"""Small on-policy PPO learner; inference artifacts are plain numeric JSON.

Only a task-owned training run may enable learning. Public delivered task
counters supply reward; neither future traces nor actual evaluation scores are
policy inputs. PPO uses recorded behavior log-probabilities and GAE by cell.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .latency_scheduler_adaptive import ACTIONS,CONTEXT_FEATURES

SCALES=[1,1,100,100,20,20,20,20,1,1,8,4,200,200,100,20,20,20,1,10,10,5,1,1,80,16,1,1,2,2]

def normalize(ctx,policy):
    means=policy.get('input_mean',[0.]*len(ctx));scales=policy.get('input_scale',SCALES)
    return [max(-10.,min(10.,(v-m)/max(1e-6,s))) for v,m,s in zip(ctx,means,scales)]

def affine(layer,x):
    return [sum(a*b for a,b in zip(row,x))+bias for row,bias in zip(layer['weight'],layer['bias'])]

def forward(policy,ctx):
    x=normalize(ctx,policy)
    for layer in policy['network']['hidden']:x=[math.tanh(v) for v in affine(layer,x)]
    logits=affine(policy['network']['actor'],x)
    value=affine(policy['network']['critic'],x)[0]
    mask=set(policy.get('allowed_actions',range(len(ACTIONS))))
    logits=[v if i in mask else -1e9 for i,v in enumerate(logits)]
    top=max(logits);exps=[math.exp(v-top) for v in logits];den=sum(exps)
    return logits,[v/den for v in exps],value

def inference(policy,ctx,cid):
    logits,probs,value=forward(policy,ctx)
    i=max(range(len(probs)),key=lambda j:probs[j])
    return i,dict(probabilities=probs,value=value),'ppo_deterministic_inference'

class PPOTrainer:
    def __init__(self,policy,log_path,seed):
        import torch
        self.torch=torch;self.policy=deepcopy(policy);self.seed=seed
        self.generator=torch.Generator(device='cpu').manual_seed(seed+701)
        self.directory=Path(log_path).parent/'ppo-checkpoints';self.directory.mkdir(exist_ok=True)
        # Linear constructors consume CPU RNG. Restore it to leave the PHY RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also seeds CUDA. Touch only the CPU generator
            # whose state this context saves/restores.
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed+703).get_state())
            self.hidden=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT_FEATURES),32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh())
            self.actor=torch.nn.Linear(32,len(ACTIONS));self.critic=torch.nn.Linear(32,1)
        net=policy.get('network')
        if net:
            for dst,src in zip([self.hidden[0],self.hidden[2],self.actor,self.critic],net['hidden']+[net['actor'],net['critic']]):
                with torch.no_grad():
                    dst.weight.copy_(torch.tensor(src['weight']));dst.bias.copy_(torch.tensor(src['bias']))
        self.parameters=list(self.hidden.parameters())+list(self.actor.parameters())+list(self.critic.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=policy.get('learning_rate',3e-4),eps=1e-5)
        self.pending={};self.rows=[];self.updates=0;self.steps=0
        self.events=Path(log_path).parent/'ppo-training.jsonl';self.refresh_policy()
        self.checkpoint('initial')

    def tensor(self,ctx):return self.torch.tensor(normalize(ctx,self.policy),dtype=self.torch.float32)

    def evaluate(self,x):
        hidden=self.hidden(x);logits=self.actor(hidden)
        allowed=set(self.policy.get('allowed_actions',range(len(ACTIONS))))
        mask=self.torch.tensor([0. if i in allowed else -1e9 for i in range(len(ACTIONS))])
        return self.torch.distributions.Categorical(logits=logits+mask),self.critic(hidden).squeeze(-1)

    @staticmethod
    def counters(users):
        keys=('offloaded_completed','offloaded_completion_latency_ms_sum','offloaded_failed','offloaded_pending')
        return [sum(u.ue_task_metrics.get(k,0) for u in users) for k in keys]

    def finish_step(self,cid,ctx,users,slot,dt):
        previous=self.pending.pop(cid,None)
        if previous is None:return
        counters=self.counters(users);delta=[a-b for a,b in zip(counters,previous['counters'])]
        if any(v< -1e-8 for v in delta[:3]):raise ValueError('Task counter reset in PPO rollout')
        elapsed_ms=(slot-previous['slot'])*dt
        reward=-(delta[1]+100.*delta[2]+elapsed_ms*counters[3])/100.
        with self.torch.no_grad():_,next_value=self.evaluate(self.tensor(ctx))
        row=dict(cid=cid,slot=previous['slot'],next_slot=slot,context=previous['context'],action=previous['action'],
            log_prob=previous['log_prob'],value=previous['value'],next_value=float(next_value),reward=reward,
            completed_delta=delta[0],latency_sum_delta_ms=delta[1],failed_delta=delta[2],pending=counters[3],policy_update=previous['update'])
        self.rows.append(row);self.steps+=1
        with self.events.open('a') as f:f.write(json.dumps(dict(kind='transition',**row),allow_nan=False)+'\n')

    def choose(self,cid,ctx,users,slot):
        with self.torch.no_grad():
            dist,value=self.evaluate(self.tensor(ctx))
            action=int(self.torch.multinomial(dist.probs,1,generator=self.generator).item())
            log_prob=float(dist.log_prob(self.torch.tensor(action)))
        self.pending[cid]=dict(context=list(ctx),action=action,log_prob=log_prob,value=float(value),
            counters=self.counters(users),slot=slot,update=self.updates)
        return action,dict(probabilities=dist.probs.tolist(),value=float(value),log_prob=log_prob,update=self.updates),'ppo_on_policy_training'

    def maybe_update(self):
        if len(self.rows)<self.policy.get('rollout_rows',32):return
        torch=self.torch;rows=self.rows;self.rows=[]
        gamma=self.policy.get('gamma',.97);lam=self.policy.get('gae_lambda',.95)
        advantages=[0.]*len(rows)
        for cid in sorted({r['cid'] for r in rows}):
            indices=[i for i,r in enumerate(rows) if r['cid']==cid]
            gae=0.
            for i in reversed(indices):
                row=rows[i];delta=row['reward']+gamma*row['next_value']-row['value']
                gae=delta+gamma*lam*gae;advantages[i]=gae
        returns=torch.tensor([a+r['value'] for a,r in zip(advantages,rows)])
        adv=torch.tensor(advantages);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        x=torch.stack([self.tensor(r['context']) for r in rows]);actions=torch.tensor([r['action'] for r in rows])
        old=torch.tensor([r['log_prob'] for r in rows]);losses=[]
        for epoch in range(self.policy.get('ppo_epochs',6)):
            order=torch.randperm(len(rows),generator=self.generator)
            for indices in order.split(self.policy.get('minibatch_rows',32)):
                dist,values=self.evaluate(x[indices]);ratio=(dist.log_prob(actions[indices])-old[indices]).exp()
                clip=self.policy.get('clip_ratio',.2)
                policy_loss=-torch.minimum(ratio*adv[indices],ratio.clamp(1.-clip,1.+clip)*adv[indices]).mean()
                value_loss=(values-returns[indices]).square().mean()
                entropy=dist.entropy().mean()
                loss=policy_loss+.5*value_loss-self.policy.get('entropy_coefficient',.015)*entropy
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite PPO loss')
                self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.parameters,.5);self.optimizer.step()
                losses.append(float(loss.detach()))
        self.updates+=1;self.refresh_policy();self.checkpoint(f'update-{self.updates:03}')
        with self.events.open('a') as f:f.write(json.dumps(dict(kind='ppo_update',update=self.updates,rows=len(rows),steps=self.steps,
            mean_reward=sum(r['reward'] for r in rows)/len(rows),mean_loss=sum(losses)/len(losses),
            actual_behavior_log_prob_used=True,gae_grouped_by_cell=True),allow_nan=False)+'\n')

    def refresh_policy(self):
        def layer(v):return dict(weight=v.weight.detach().tolist(),bias=v.bias.detach().tolist())
        self.policy['network']=dict(hidden=[layer(self.hidden[0]),layer(self.hidden[2])],actor=layer(self.actor),critic=layer(self.critic))
        self.policy['training_updates']=self.updates;self.policy['training_steps']=self.steps

    def checkpoint(self,name):
        value=deepcopy(self.policy);value['training']=False
        p=self.directory/(name+'.json');p.write_text(json.dumps(value,allow_nan=False)+'\n')
        tmp=self.directory/'latest.tmp';tmp.write_text(json.dumps(value,allow_nan=False)+'\n');tmp.replace(self.directory/'latest.json')
        # Optimizer and RNG are kept separately; deployed policies only read JSON.
        self.torch.save(dict(optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),
            update=self.updates,steps=self.steps),self.directory/(name+'-optimizer.pt'))

"""Durable user-authored preset text/memos, separate from intent constraints."""
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
from smo.intent_language import IntentError

IDS=frozenset(('ue_power_mec_budget_band','ue_power_mec_delay_band','bs_power_mec_delay_band','c04','c06','c02','i16','lc02','c03'))


class PresetConflict(IntentError):pass


def validate(edits,deleted):
    if not isinstance(edits,dict)or set(edits)-IDS or not isinstance(deleted,list)or any(not isinstance(v,str)or v not in IDS for v in deleted):
        raise IntentError('预设 ID 或删除记录无效。')
    normalized={}
    for ident,entry in edits.items():
        if not isinstance(entry,dict)or set(entry)-{'text','tag'}or not isinstance(entry.get('text'),str)or not isinstance(entry.get('tag',''),str):
            raise IntentError('预设只接受意图文字和自由备注 tag。')
        text,tag=entry['text'].strip(),entry.get('tag','').strip()
        if not text or len(text)>2000 or len(tag)>120:raise IntentError('意图须为 1–2000 字符，tag 最多 120 字符。')
        normalized[ident]=dict(text=text,tag=tag)
    return normalized,list(dict.fromkeys(deleted))


class PresetStore:
    def __init__(self,path=None):
        self.path=Path(path)if path else None;self.lock=RLock()
        self.data=dict(schema='oran.intent-presets.v2',revision=0,edits={},deleted_ids=[])
        if self.path and self.path.exists():
            data=json.loads(self.path.read_text())
            if data.get('schema')!='oran.intent-presets.v2'or type(data.get('revision'))is not int:raise ValueError('Invalid preset store')
            edits,deleted=validate(data['edits'],data['deleted_ids'])
            self.data=dict(schema='oran.intent-presets.v2',revision=data['revision'],edits=edits,deleted_ids=deleted)

    def read(self):
        with self.lock:return deepcopy(self.data)

    def save(self,revision,edits,deleted):
        edits,deleted=validate(edits,deleted)
        with self.lock:
            if type(revision)is not int or revision!=self.data['revision']:
                raise PresetConflict('预设已在另一个窗口更新。请刷新页面后再保存；你的输入仍保留在编辑框中。')
            if edits==self.data['edits']and deleted==self.data['deleted_ids']:return self.read()
            updated=dict(schema='oran.intent-presets.v2',revision=revision+1,edits=edits,deleted_ids=deleted)
            if self.path:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.with_suffix('.history.jsonl').open('a')as history:history.write(json.dumps(self.data,ensure_ascii=False)+'\n')
                tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n');tmp.replace(self.path)
            self.data=updated
            return self.read()

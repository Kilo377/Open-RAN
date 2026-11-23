"""Prototype index for a factory-published request-history snapshot.

The official manager owns membership and immutable string request IDs after
publication; status/reason/updated_at_slot remain mutable. Generic resources
without this explicit factory index retain the original linear lookup.

The factory owner maintains first positions incrementally, giving O(1)
lookups. The diagnostic per-snapshot fallback builds an O(n) first index.
Publishing the unchanged full history list is still O(n) per slot, so this
does NOT remove all quadratic lifetime work from the surrounding simulator.
"""
from models.oran_lifecycle import E2ControlRequest


class FactoryRequestStore(dict):
    """Canonical manager-owned, append-order index; never removes history.

    Valid only for immutable standard-string IDs exactly equal to dict keys.
    Unsupported external writes revoke the capability; old snapshots fall back.
    Replacing an object at the same key preserves its position, and snapshots
    still update their own original object rather than the replacement.
    """
    def __init__(self):
        super().__init__();self.first_positions={};self.trusted=True

    def __setitem__(self,key,value):
        if type(key)is not str or type(value)is not E2ControlRequest or type(value.request_id)is not str or value.request_id!=key:
            self.trusted=False
        if key not in self:self.first_positions[key]=len(self)
        super().__setitem__(key,value)

    def __delitem__(self,key):self.trusted=False;super().__delitem__(key)
    def clear(self):self.trusted=False;super().clear()
    def pop(self,*args):self.trusted=False;return super().pop(*args)
    def popitem(self):self.trusted=False;return super().popitem()
    def update(self,*args,**kwargs):
        for key,value in dict(*args,**kwargs).items():self[key]=value
    def setdefault(self,key,default=None):
        if key not in self:self[key]=default
        return self[key]
    def __ior__(self,other):self.update(other);return self


def index_published_requests(resources,owner=None,*,snapshot_index=False):
    history=resources.e2_control_requests
    if (type(history)is list and type(owner)is FactoryRequestStore and owner.trusted
            and len(history)==len(owner)):
        resources._published_request_owner=(history,len(history),owner)
        return resources
    if not snapshot_index:return resources
    # Do not eagerly stringify arbitrary malformed/external objects: doing so
    # could raise earlier than the original first-match lookup would.
    if type(history)is not list or any(type(r)is not E2ControlRequest
            or type(r.request_id)is not str for r in history):
        return resources
    first={}
    for i,request in enumerate(history):
        first.setdefault(request.request_id,i)
    resources._published_request_lookup=(history,len(history),first)
    return resources


def published_request(resources,request_id):
    """(usable, object|None). IDs/membership must obey the factory contract."""
    managed=getattr(resources,'_published_request_owner',None)
    if managed is not None:
        history,length,owner=managed
        if resources.e2_control_requests is not history or len(history)!=length or not owner.trusted:
            return False,None
        position=owner.first_positions.get(request_id)
        if position is None or position>=length:return True,None
        request=history[position]
        if type(request)is E2ControlRequest and type(request.request_id)is str and request.request_id==request_id:
            return True,request
        return False,None
    index=getattr(resources,'_published_request_lookup',None)
    if index is None:return False,None
    history,length,first=index
    if resources.e2_control_requests is not history or len(history)!=length:
        return False,None
    position=first.get(request_id)
    if position is None:return True,None
    request=history[position]
    if type(request)is not E2ControlRequest or type(request.request_id)is not str or request.request_id!=request_id:
        return False,None
    return True,request

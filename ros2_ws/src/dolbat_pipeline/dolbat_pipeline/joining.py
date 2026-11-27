"""Bounded exact-frame join; timestamps and indexes must agree across stages."""


class FrameJoiner:
    def __init__(self,kinds=('video','debug','path','control'),max_pending=16):
        self.kinds=frozenset(kinds);self.max_pending=max_pending
        self.next_index=0;self.pending={};self.stamps={}

    def add(self,kind,index,stamp,value):
        if kind not in self.kinds:raise ValueError('Unknown frame stream')
        index=int(index)
        if index<self.next_index:return []
        if index>=self.next_index+self.max_pending:raise ValueError('Frame cache limit exceeded; a stream may be missing')
        stamp=tuple(stamp)
        if index in self.stamps and self.stamps[index]!=stamp:raise ValueError('Frame timestamp mismatch')
        self.stamps[index]=stamp
        group=self.pending.setdefault(index,{})
        if kind in group:raise ValueError('Duplicate frame before synchronization')
        group[kind]=value;ready=[]
        while self.next_index in self.pending and self.kinds<=self.pending[self.next_index].keys():
            i=self.next_index;ready.append((i,self.pending.pop(i)));self.stamps.pop(i)
            self.next_index+=1
        return ready

    def finish(self,total):
        if self.next_index!=total or self.pending:raise ValueError('Incomplete synchronized recording')

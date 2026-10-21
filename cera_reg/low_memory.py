"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import types, torch
from einops import rearrange


@torch.no_grad()
def forward(self,input_q,input_k,input_v,embed_qk,key_weights=None,key_masks=None,attention_factors=None):
 if self.training:raise RuntimeError('Low-memory RPE is inference-only')
 q=rearrange(self.proj_q(input_q),'b n (h c) -> b h n c',h=self.num_heads)
 k=rearrange(self.proj_k(input_k),'b m (h c) -> b h m c',h=self.num_heads)
 v=rearrange(self.proj_v(input_v),'b m (h c) -> b h m c',h=self.num_heads)
 hidden=[];scores=[]
 for start in range(0,input_q.shape[1],32):
  stop=start+32;qq=q[:,:,start:stop]
  p=rearrange(self.proj_p(embed_qk[:,start:stop]),'b n m (h c) -> b h n m c',h=self.num_heads)
  a=(torch.einsum('bhnc,bhmc->bhnm',qq,k)+torch.einsum('bhnc,bhnmc->bhnm',qq,p))/self.d_model_per_head**.5
  if attention_factors is not None:a=a*attention_factors[:,start:stop].unsqueeze(1)
  if key_weights is not None:a=a*key_weights.unsqueeze(1).unsqueeze(1)
  if key_masks is not None:a=a.masked_fill(key_masks.unsqueeze(1).unsqueeze(1),float('-inf'))
  a=self.dropout(torch.softmax(a,dim=-1));hidden.append(torch.matmul(a,v));scores.append(a)
 return rearrange(torch.cat(hidden,dim=2),'b h n c -> b n (h c)'),torch.cat(scores,dim=2)


def install(model):
 from geotransformer.modules.transformer.rpe_transformer import RPEMultiHeadAttention
 n=0
 for m in model.modules():
  if isinstance(m,RPEMultiHeadAttention):m.forward=types.MethodType(forward,m);n+=1
 return n


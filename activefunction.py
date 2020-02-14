#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as np
def xyplot(x_vals, y_vals, name):
    # d2l.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')


# In[44]:


y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


# In[45]:


y = x.sigmoid()
xyplot(x, y, 'sigmoid')


# In[46]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


# In[47]:


y = x.tanh()
xyplot(x, y, 'tanh')


# In[48]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')


# In[ ]:





import mytorch

# batch of 2 samples
x = mytorch.Tensor([1,2,3,4],[2,2],True)
b = mytorch.Tensor([10,20],[1,2],True)

y = mytorch.add(x,b)
loss = mytorch.sum(y)

loss.backward()

print("b.grad:",b.grad)

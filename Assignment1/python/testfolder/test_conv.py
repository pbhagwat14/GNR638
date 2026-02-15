import mytorch

x = mytorch.Tensor([1]* (1*1*6*6),[1,1,6,6],False)

conv = mytorch.Conv2D(1,1,3,3,0,2)
y = conv.forward(x)

print(y.shape)

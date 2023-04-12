# importance test
ximp = torch.linspace(0,1,11).reshape(-1, 1)
timp = torch.zeros(11).reshape(-1, 1)
ximp.requires_grad = True
timp.requires_grad = True
importance = torch.abs(criterion.differential_loss(model, timp, ximp))
importance *= 1/torch.sum(importance)
ximp_weighted = importance * ximp

# plot attmepted solution
x = torch.Tensor(np.linspace(0, 1, 1000).reshape(-1, 1))
t = torch.Tensor(np.ones(1000).reshape(-1, 1))
plt.plot(
    x.numpy(),
    criterion.u0(x).detach().numpy(),
    label="ground truth t = 0",
)
plt.plot(x.numpy(), model(0 * t, x).detach().numpy(), label="predicted t = 0")
plt.plot(ximp.detach().numpy(), model(timp, ximp).detach().numpy(), 'o', label='sample points')
plt.plot(ximp_weighted.detach().numpy(), model(timp, ximp_weighted).detach().numpy(), 'o', label='importance weighted')
plt.legend()
plt.show()
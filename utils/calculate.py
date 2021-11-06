import torch as tc


def psnr(out, hr):
    with tc.no_grad():
        criteria = tc.nn.MSELoss()
        mse = criteria(out, hr)
        return -10 * tc.log10(mse)


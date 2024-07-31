from torch.optim import lr_scheduler


# ReduceLROnPlateau, step() used after validation split on val_loss
scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1,
                                           patience=10, threshold=0.0001,
                                           cooldown=0, min_lr=0)
scheduler.step(val_loss)

# labda scheduler, step() used at the end of each epoch
lambda1 = lambda epoch: 0.65 ** epoch
scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
scheduler.step()


# simple scheduler, decays lr with each step
scheduler = lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
scheduler.step()

# simple scheduler, decays lr when milestone epoch is hit
scheduler = lr_scheduler.MultiStepLR(opt, milestones=[6,8,9], gamma=0.1)
scheduler.step()
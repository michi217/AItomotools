def save_loss_values_in_txt(savepath, epoch, loss_value):
    with open(savepath + '/loss.txt', 'a+') as f:
        f.write(str(epoch) + ' ' + str(round(loss_value, 2)) + '\n')
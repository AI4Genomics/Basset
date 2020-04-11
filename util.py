import datetime


def onehots_to_charmap(onehots, rev_charmap, generic=True):
    """Function to transfer a onehoted sequence to its equivalent charmaps"""
    st = ""
    onehots_transposed = zip(*onehots)
    for _ in onehots_transposed:
        st = st + rev_charmap.get(np.argmax(_), "-")
    return st


def charmap_to_onehots(chars, charmap):
    """Function to transfer a charmaps to its equivalent onehoted sequence"""
    onehots = np.int_(np.zeros([len(chars), vocab_size]))
    for i, char in enumerate(chars):
        onehots[i] = I[charmap[char]]
    return onehots


def cal_iter_time(former_iteration_endpoint, tz):
    """Calculating 'Computation Time' for this round of iteration"""
    current_time = datetime.datetime.now(tz)
    time_elapsed = current_time - former_iteration_endpoint
    #print(" ~~  Time current: {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
    print("~~~ Time elapsed for this epoch: {} ~~~\n\n".format(str(time_elapsed).split(".")[0]))
    return current_time


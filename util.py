def onehots_to_charmap(onehots, rev_charmap, model_type="resnet2d", generic=True):
    """Function to transfer a onehoted sequence to its equivalent charmaps"""
    st = ""
    if model_type == "mlp":
        length = len(rev_charmap)
        onehots = np.reshape(onehots, [-1, length])
        for _ in onehots:
            st = st + rev_charmap.get(np.argmax(_), "-")
        return st
    if model_type == "resnet1d" or model_type == "resnet2d":
        onehots_transposed = zip(*onehots)
        for _ in onehots_transposed:
            st = st + rev_charmap.get(np.argmax(_), "-")
        return st


def charmap_to_onehots(chars, charmap, model_type="resnet2d"):
    """Function to transfer a charmaps to its equivalent onehoted sequence"""
    onehots = np.int_(np.zeros([len(chars), vocab_size]))
    for i, char in enumerate(chars):
        onehots[i] = I[charmap[char]]

    if model_type == "mlp":
        onehots = np.reshape(onehots, [-1])
    if model_type == "resnet1d" or model_type == "resnet2d":
        onehots = onehots.T
    return onehots
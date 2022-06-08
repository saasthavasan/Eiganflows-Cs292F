def calc_accuracy(outputs, batch_labels):
    correct = 0
    total = 0
    for b_labels, output in zip(batch_labels, outputs):

        for label, out in zip(b_labels, output):
            val = 0
            # import IPython
            # IPython.embed()
            # assert False
            if float(out)> 0.5:
                val = 1
            if val == int(label):
                correct += 1
            total += 1

    # accuracy = float(correct)/ total
    return correct, total



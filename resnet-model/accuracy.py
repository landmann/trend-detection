def accuracy(output, target, ks=(1,), ):
    maxk = max(ks) 
    batch_size = target.size(0)
    # output is a batch_size x num_categories tensor (row vecs)
    # returns the indices (aka cat ids)
    _, prediction = output.topk(maxk, dim=1, largest=True, sorted=True)
    # prediction is a batch_size x k array where each row is the top k index predictions for that batch item.
    prediction = prediction.t() # transpose to a matrix of dim k x batch_size
    # the view call reshapes the tensor w/ one row and arbitrary cols 
    # expand_as just drags out the end of the tensor out....
    #this line flattens the target into a row vector with len bach_size and repeats it k times to get something the same shape as prediction, then equates
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    ret = []
    for k in ks: 
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        ret.append(correct_k.mul(100.0 / batch_size))
    return ret



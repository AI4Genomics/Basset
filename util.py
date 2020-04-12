import datetime

def cal_iter_time(former_iteration_endpoint, tz):
    """Calculating 'Computation Time' for this round of iteration"""
    current_time = datetime.datetime.now(tz)
    time_elapsed = current_time - former_iteration_endpoint
    #print(" ~~  Time current: {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
    print("~~~ Time elapsed for this epoch: {} ~~~\n\n".format(str(time_elapsed).split(".")[0]))
    return current_time


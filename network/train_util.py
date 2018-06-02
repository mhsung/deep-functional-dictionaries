# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

from clr_gurobi import CLR
import math
import numpy as np
import os
import sys
import tensorflow as tf
import time


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def validate(sess, net, exp_type, data):
    count = 0
    loss = 0.0
    summary = None

    if exp_type == 'ours':
        clr = CLR(net.batch_size, net.K)

    for P, b, L in data:
        if exp_type == 'ours':
            AtA, btA = sess.run([net.AtA, net.btA], feed_dict={
                net.P: P, net.b: b, net.is_training: False})
            x = clr.solve(AtA, btA)
            feed_dict = {net.P: P, net.b: b, net.x: x}
        elif exp_type == 'sem_seg':
            feed_dict = {net.P: P, net.b: b, net.L: L}
        else:
            assert(False)

        # NOTE:
        # Take the summary of the last random batch.
        feed_dict[net.is_training] = False
        summary, step_loss = sess.run([
            net.summary, net.loss], feed_dict=feed_dict)

        count += data.step_size
        loss += (step_loss * data.step_size)

    loss /= float(count)
    return loss, summary


def train(sess, net, exp_type, train_data, val_data, n_epochs, snapshot_epoch,
        validation_epoch, model_dir='model', log_dir='log', data_name='',
        output_generator=None):
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Create snapshot directory.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print ("\n=================")
    print ("Training started.")

    start_time = time.time()
    clear_prev_line = False

    if exp_type == 'ours':
        clr = CLR(net.batch_size, net.K)

    for epoch in range(1, n_epochs + 1):
        for P, b, L in train_data:
            if exp_type == 'ours':
                AtA, btA = sess.run([net.AtA, net.btA], feed_dict={
                    net.P: P, net.b: b, net.is_training: False})
                x = clr.solve(AtA, btA)
                feed_dict = {net.P: P, net.b: b, net.x: x}
            elif exp_type == 'sem_seg':
                feed_dict = {net.P: P, net.b: b, net.L: L}
            else:
                assert(False)

            feed_dict[net.is_training] = True
            step, _, = sess.run([net.global_step, net.train_op],
                    feed_dict=feed_dict)

            feed_dict[net.is_training] = False
            summary, loss = sess.run([
                net.summary, net.loss], feed_dict=feed_dict)

            if step > 1000:
                train_writer.add_summary(summary, step)

            elapsed = time.time() - start_time
            msg = " -"
            msg += "" if data_name == '' else " [{}]".format(data_name)
            msg += " Step: {:d}".format(step)
            msg += " | Iter {:d}/{:d}".format(
                    train_data.end, train_data.n_data)
            msg += " | Batch Loss: {:6f}".format(loss)
            msg += " | Elapsed Time: {}".format(hms_string(elapsed))

            if clear_prev_line: sys.stdout.write("\033[1A[\033[2K")
            print(msg)
            clear_prev_line = True

        if epoch % validation_epoch == 0:
            # Calculate total train and validation loss.
            loss, _ = validate(sess, net, exp_type, train_data)
            msg = "||"
            msg += "" if data_name == '' else " [{}]".format(data_name)
            msg += " Epoch: {:d}".format(epoch)
            msg += " | Train Loss: {:6f}".format(loss)

            val_loss, val_summary = validate(
                    sess, net, exp_type, val_data)
            msg += " | Valid Loss: {:6f}".format(val_loss)

            if step > 1000:
                test_writer.add_summary(val_summary, step)

            elapsed = time.time() - start_time
            remaining = elapsed / epoch * (n_epochs - epoch)
            msg += " | Elapsed Time: {} | Remaining Time: {} ||".format(
                    hms_string(elapsed), hms_string(remaining))

            if clear_prev_line: sys.stdout.write("\033[1A[\033[2K")
            print(msg)
            clear_prev_line = False

        if epoch % snapshot_epoch == 0:
            # Save snapshot.
            if clear_prev_line: sys.stdout.write("\033[1A[\033[2K")
            sys.stdout.write("Saving epoch {:d} snapshot... ".format(epoch))
            net.saver.save(sess, model_dir + '/tf_model.ckpt',
                    global_step=step)
            print('Done.')
            clear_prev_line = False

            # Generate outputs.
            if output_generator is not None:
                output_generator(sess, 'snapshot_{:06d}'.format(epoch))

    train_writer.close()
    test_writer.close()

    elapsed = time.time() - start_time
    print ("Training finished.")
    print (" - Elapsed Time: {}".format(hms_string(elapsed)))
    print ("Saved '{}'.".format(
        net.saver.save(sess, model_dir + '/tf_model.ckpt', global_step=step)))
    print ("=================\n")


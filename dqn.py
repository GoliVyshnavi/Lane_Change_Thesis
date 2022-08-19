import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import datetime
import os
import random
import math
from rm import ReplayMemory

# reference-----https://github.com/Ilhem23/change_lane_DQN

def Veh_model(no_action, input_size):
    with tf.device('/gpu:0'):
        input = kl.Input(shape=(input_size))
        h1 = kl.Dense(64, activation='relu')(input)
        h2 = kl.Dense(128, activation='relu')(h1)
        h3 = kl.Dense(64, activation='relu')(h2)
        states = kl.Dense(1)(h3)
        states = kl.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1), output_shape=(no_action,))(
            states)

        action_adv = kl.Dense(no_action)(h3)
        action_adv = kl.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),
                                     output_shape=(no_action,))(
            action_adv)

        xx = kl.Add()([states, action_adv])

        agent = tf.keras.models.Model(input, xx, name='CarModel')
        return agent



class DQNAgent:
    def __init__(self, func=None, alpha=0.00001, discount=0.99, batch_size=32):
        self.discount = discount
        self.learning_rate = alpha
        self.crnt_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.snapshot_dir = 'checkpoints/'
        self.agent_name = 'DQN_test'
        self.agent_direc = self.snapshot_dir + self.agent_name
        self.logs_direc = 'logs/'
        self.log_training_direc = self.logs_direc + self.agent_name
        self.create_logs_directory()
        self.tr_summary_writer = tf.summary.create_file_writer(self.log_training_direc)
        self.func = func
        self.eps_strt = 0.9
        self.eps_stop = 0.5
        self.terminated_steps = 0
        self.eps_decay = 100
        self.batch_size = batch_size
        self._tau = 0.08
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.primary_nw = Veh_model(no_action=3, input_size=37)
        self.target_nw = Veh_model(no_action=3, input_size=37)

    def create_logs_directory(self):
        if not os.path.exists(self.logs_direc):
            os.mkdir(self.logs_direc)
        if not os.path.exists(self.log_training_direc):
            os.mkdir(self.log_training_direc)
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        if not os.path.exists(self.agent_direc):
            os.mkdir(self.agent_direc)

    def action(self, state, primary_nw):
        ep_limit = self.eps_stop + (self.eps_strt - self.eps_stop) * \
                        math.exp(-1. * self.terminated_steps / self.eps_decay)

        if np.random.rand() < ep_limit:
            a = random.randint(0, 2)
        else:
            a = primary_nw.predict(np.expand_dims(state, axis=0))
            a = np.argmax(a)
        return a

    def training_steps(self, exp_replay_mem):
        state, action, reward, nw_state, term_flag = exp_replay_mem.get_minibatch()
        q_values = self.primary_nw(nw_state)
        q_values=q_values.numpy()
        nw_actions = np.argmax(q_values, axis=1)
        q_values = self.target_nw(nw_state)
        q_values = np.array([q_values[no, action_a] for no, action_a in enumerate(nw_actions)])
        final_q = reward + (self.discount * q_values * (1 - term_flag))
        l = self.primary_nw.train_on_batch(state, final_q)
        return l

    def modify_network(self):
        for i, j in zip(self.primary_nw.trainable_variables, self.target_nw.trainable_variables):
            i.assign(i * (1 - self._tau) + j * self._tau)

    def train(self, env, steps_in_epoch=128, num_epochs=2500):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        with tf.device('/gpu:0'):
            freq_update = 4
            freq_update_nw = 2500
            STRT_SIZE_REPLAY_MEM = 33
            self.primary_nw.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
            replay_mem = ReplayMemory()
            avg_loss = tf.keras.metrics.Mean()
            tot_training_rew = tf.keras.metrics.Mean(name='mean', dtype=None)
            tot_training_rew_comf = tf.keras.metrics.Mean(name='mean', dtype=None)
            tot_training_rew_eff = tf.keras.metrics.Mean(name='mean', dtype=None)
            tot_training_rew_safe = tf.keras.metrics.Mean(name='mean', dtype=None)
            training_rew_coll = tf.keras.metrics.Mean()
            training_rew_speed = tf.keras.metrics.Mean()

            upcoming_obs = env.reset(gui=True, no_Vehicle=35)
            epoch_1 = 0
            try:
                for ep in range(epoch_1, num_epochs):
                    print('Epoch ',ep)
                    ep_rewards = 0
                    for step in range(steps_in_epoch):
                        state = upcoming_obs.copy()
                        action = self.action(state, self.primary_nw)
                        upcoming_obs, arr_rewards, terminal, collision_flag = env.new_step(action)
                        avg_spd_pc = env.speed / env.max_speed
                        tot_rew, tot_comf, tot_eff, tot_safe = arr_rewards
                        # Adding experience to replay memory

                        replay_mem.add_experience(action=action,
                                                        frame=upcoming_obs,
                                                        reward=tot_rew,
                                                        terminal=terminal)
                       
                        tot_training_rew.update_state(tot_rew)
                        tot_training_rew_comf.update_state(tot_comf)
                        tot_training_rew_eff.update_state(tot_eff)
                        tot_training_rew_safe.update_state(tot_safe)
                        training_rew_coll.update_state(collision_flag)
                        training_rew_speed.update_state(avg_spd_pc)

                        if self.terminated_steps > STRT_SIZE_REPLAY_MEM:
                            loss = self.training_steps(replay_mem)
                            avg_loss.update_state(loss)
                            self.modify_network()
                        else:
                            avg_loss.update_state(-1)

                        if step % freq_update_nw == 0 and step > STRT_SIZE_REPLAY_MEM:
                            self.target_nw.set_weights(self.primary_nw.get_weights())

                        self.terminated_steps += 1

                    with self.tr_summary_writer.as_default():
                        tf.summary.scalar('loss of model', avg_loss.result(), step=ep)
                        tf.summary.scalar('total rewards', tot_training_rew.result(), step=ep)
                        tf.summary.scalar('comfort rewards',  tot_training_rew_comf.result(), step=ep)
                        tf.summary.scalar('efficiency rewards', tot_training_rew_eff.result(), step=ep)
                        tf.summary.scalar('safety rewards', tot_training_rew_safe.result(), step=ep)
                        tf.summary.scalar('collision', training_rew_coll.result(), step=ep)
                        tf.summary.scalar('avg speed vs target speed', training_rew_speed.result(), step=ep)

                    tot_training_rew.reset_states()
                    tot_training_rew_comf.reset_states()
                    tot_training_rew_eff.reset_states()
                    tot_training_rew_safe.reset_states()
                    training_rew_coll.reset_states()
                    training_rew_speed.reset_states()

                    # Saving file of model
                    if ep % 250 == 0:
                        tf.keras.models.save_model(self.primary_nw,
                                                   self.agent_direc + "/" + str(ep) + "primary_network.hp5",
                                                   save_format="h5")
                        tf.keras.models.save_model(self.target_nw,
                                                   self.agent_direc + "/" + str(ep) + "target_nw.hp5",
                                                   save_format="h5")
            except KeyboardInterrupt:
                tf.keras.models.save_model(self.primary_nw, self.agent_direc + "/" + str(ep) + "primary_network.hp5",
                                           save_format="h5")
                tf.keras.models.save_model(self.target_nw,
                                           self.agent_direc + "/" + str(ep) + "target_nw.hp5", save_format="h5")

            env.close()

            return 0


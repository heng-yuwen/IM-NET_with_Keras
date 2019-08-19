import os
import time
import math
from glob import glob # Use Regular Expression to find file paths.
import tensorflow as tf
import numpy as np
import h5py # Read h5 file(the data of the 3D models).
import cv2
import mcubes # Marching cubes algorithm.

from ops import *

class IMAE(object):
	def __init__(self, sess, real_size, batch_size_input, is_training = False, z_dim=128, ef_dim=32, gf_dim=128, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
			is_training (bool): Control the network is now training or inferencing.
			z_dim (int): The dimension of the latent space.
			real_size (int): The output voxel grid size.
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16*2)
		#3-- (64, 32*32*32)
		#4-- (128, 32*32*32*4)
		self.real_size = real_size #output point-value voxel grid size in training, 64
		# ??It seems that batch size is the number of points in a 3D model. i.e. real_size*real_size*real_size.
		self.batch_size_input = batch_size_input #training batch size (virtual, batch_size is the real batch_size), 32768=32*32*32

		self.batch_size = 16*16*16*4 #adjust batch_size according to gpu memory size in training(Uppder limit, 16384)

		# If input size is smaller than the batch size, we just use one batch.
		if self.batch_size_input<self.batch_size:
			self.batch_size = self.batch_size_input

		self.input_size = 64 #input voxel grid size

		self.z_dim = z_dim
		self.ef_dim = ef_dim # Encoder filters
		self.gf_dim = gf_dim # 128

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir

		if os.path.exists(self.data_dir+'/'+self.dataset_name+'.hdf5'):
			self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'.hdf5', 'r') # Read in the hdf5 file as readonly mode.
			self.data_points = self.data_dict['points_'+str(self.real_size)][:]
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_voxels = self.data_dict['voxels'][:]
			if self.batch_size_input!=self.data_points.shape[1]:
				print("error: batch_size!=data_points.shape")
				exit(0)
			if self.input_size!=self.data_voxels.shape[1]:
				print("error: input_size!=data_voxels.shape")
				exit(0)
		else:
			if is_training:
				print("error: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				exit(0)
			else:
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')

		if not is_training:
			self.real_size = 64 #output point-value voxel grid size in testing
			self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
			self.batch_size = self.test_size*self.test_size*self.test_size #do not change

			#get coords
			dima = self.test_size # 32
			dim = self.real_size # 64

			# Use test_size to initialize grids but don't know for what use???.
			self.aux_x = np.zeros([dima,dima,dima],np.uint8) # 32*32*32
			self.aux_y = np.zeros([dima,dima,dima],np.uint8)
			self.aux_z = np.zeros([dima,dima,dima],np.uint8)

			# For multiple scale, reduce the size accordingly.
			multiplier = int(dim/dima) # 2
			multiplier2 = multiplier*multiplier # 4
			multiplier3 = multiplier*multiplier*multiplier # 8

			# Sample the coordinates with value range(0, 64, 2) for x, y, z.
			for i in range(dima):
				for j in range(dima):
					for k in range(dima):
						# This is actually a rotate of the 3D voxel.
						self.aux_x[i,j,k] = i*multiplier # Along width, the value is: 0, 2, 4, 6, 8, 10, 12, 14, 16,... 32,... 64
						self.aux_y[i,j,k] = j*multiplier # Along height...
						self.aux_z[i,j,k] = k*multiplier # Along channel...
			self.coords = np.zeros([multiplier3,dima,dima,dima,3], np.float32) # 8*32*32*32*3
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						# Initialize the coords with the rotation.
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i #
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k

			# coords actually contains the whole coords of the real_size cube. 64*64*64 positions.
			self.coords = (self.coords+0.5)/dim*2.0-1.0
			self.coords = np.reshape(self.coords,[multiplier3,self.batch_size,3]) # 8*32768(number of voxels)*3(x, y, z points)

		self.build_model()

	def build_model(self):
		self.vox3d = tf.placeholder(shape=[1,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32) # 1(batch)*64*64*64*1
		self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32) # 1*128
		self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32) # Number of batch_size x, y, z points
		self.point_value = tf.placeholder(shape=[self.batch_size,1], dtype=tf.float32) # The value of the voxel points.

		# The voxel encoder and IM generator(for points in point_coord).
		self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G = self.generator(self.point_coord, self.E, phase_train=True, reuse=False)

		# sG: Same as E, for AE test use. zG: for generative use.
		self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.sG = self.generator(self.point_coord, self.sE, phase_train=False, reuse=True)
		self.zG = self.generator(self.point_coord, self.z_vector, phase_train=False, reuse=True)

		# Point_value is the ground truth of the IM value(0, 1, inside or outside).
		self.loss = tf.reduce_mean(tf.square(self.point_value - self.G))

		self.saver = tf.train.Saver(max_to_keep=10)


	def generator(self, points, z, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()

			zs = tf.tile(z, [self.batch_size,1]) # batch_size*128, repeat the 1*128 vector batch_size times.
			pointz = tf.concat([points,zs],1) # batch_size*3 concate batch_size*128 = batch_size*131.
			print("pointz",pointz.shape)

			h1 = lrelu(linear(pointz, self.gf_dim*16, 'h1_lin')) # 131 to 2048
			h1 = tf.concat([h1,pointz],1)

			h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin')) # 2048 to 1024
			h2 = tf.concat([h2,pointz],1)

			h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin')) # 1024 to 512
			h3 = tf.concat([h3,pointz],1)

			h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin')) # 512 to 256
			h4 = tf.concat([h4,pointz],1)

			h5 = lrelu(linear(h4, self.gf_dim, 'h7_lin')) # 256 to 128
			h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin')) # 128 to 1

			return tf.reshape(h6, [self.batch_size,1])

	def encoder(self, inputs, phase_train=True, reuse=False):
		# Encode one voxel grid to 1*128 vector.
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()

			d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.ef_dim], strides=[1,2,2,2,1], scope='conv_1')
			d_1 = lrelu(batch_norm(d_1, phase_train))

			d_2 = conv3d(d_1, shape=[4, 4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,2,1], scope='conv_2')
			d_2 = lrelu(batch_norm(d_2, phase_train))

			d_3 = conv3d(d_2, shape=[4, 4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,2,1], scope='conv_3')
			d_3 = lrelu(batch_norm(d_3, phase_train))

			d_4 = conv3d(d_3, shape=[4, 4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,2,1], scope='conv_4')
			d_4 = lrelu(batch_norm(d_4, phase_train))

			d_5 = conv3d(d_4, shape=[4, 4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = tf.nn.sigmoid(d_5)

			return tf.reshape(d_5,[1,self.z_dim])

	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())

		# What is the differences between batch_idxs and batch_size_input???
		# Batch_idxs is the model batch. batch_size_input is the point batch.
		batch_idxs = len(self.data_points)
		batch_index_list = np.arange(batch_idxs)

		# We get 32768 points for one model(or kind?), then seperate them to two batches.
		batch_num = int(self.batch_size_input/self.batch_size) # 32768 / 16384 = 2
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)

		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in range(counter, config.epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(0, batch_idxs):
				for minib in range(batch_num):
					dxb = batch_index_list[idx] # Choose one model.
					batch_voxels = self.data_voxels[dxb:dxb+1]
					batch_points_int = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size] # Choose one batch points from model dxb. x, y, z coordinates.
					batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0 # Why?????
					batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size] # Get the ground truth value of the x, y, z coordinates.

					# Update AE network
					_, errAE = self.sess.run([ae_optim, self.loss],
						feed_dict={
							self.vox3d: batch_voxels,
							self.point_coord: batch_points,
							self.point_value: batch_values,
						})
					avg_loss += errAE
					avg_num += 1
					if (idx%512 == 0):
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, avgloss: %.8f" % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errAE, avg_loss/avg_num))

				# For the last model, use it as a inference sample.
				if idx==batch_idxs-1:
					model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32) #64*64*64
					real_model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32) #64*64*64
					for minib in range(batch_num):
						dxb = batch_index_list[idx]
						batch_voxels = self.data_voxels[dxb:dxb+1]
						batch_points_int = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
						batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]

						model_out = self.sess.run(self.sG, # model_out is the model decoded.
							feed_dict={
								self.vox3d: batch_voxels,
								self.point_coord: batch_points,
							})

						# Put voxel value to the corresponding position(For 64*64*64 model, we use 32768 points).
						model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size]) # Put	16384 points to the voxel grid.
						real_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])
					# Because we only have value (0, 1), so we project the image from three directions. And the max value across the axis, will give us the silhouette.
					img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1t.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2t.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3t.png",img3)
					img1 = np.clip(np.amax(real_model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(real_model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(real_model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1i.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2i.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3i.png",img3)
					print("[sample]")

				if idx==batch_idxs-1:
					self.save(config.checkpoint_dir, epoch)

	def test_interp(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		# Between the two z vector, we sample 8 interpolations.
		interp_size = 8
		idx1 = 0
		idx2 = 2

		# Choose two voxels.
		batch_voxels1 = self.data_voxels[idx1:idx1+1]
		batch_voxels2 = self.data_voxels[idx2:idx2+1]

		model_z1 = self.sess.run(self.sE,
			feed_dict={
				self.vox3d: batch_voxels1,
			})
		model_z2 = self.sess.run(self.sE,
			feed_dict={
				self.vox3d: batch_voxels2,
			})

		batch_z = np.zeros([interp_size,self.z_dim], np.float32) # 8*128

		# Use model_z1 and model_z2 to fill in the batch_z(8 linearly interpolated z vectors, include z1 and z2.).
		# Because gradient is fixed, so we need increase 7 times to get z2 from z1.
		for i in range(interp_size):
			batch_z[i] = model_z2*i/(interp_size-1) + model_z1*(interp_size-1-i)/(interp_size-1)


		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		for t in range(interp_size):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32) # 66*66*66
			# Get all the points for 64*64*64 space points.
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: self.coords[minib], # Get 32*32*32*3 points.
							})

						# Here, we fill the 8*32*32*32 space points back into the 64*64*64 cube just the same as how we build coords but with one more padding to make sure the surface won't touch the border.
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

			thres = 0.5
			# Get the mesh object using dicision boundry 0.5.
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+"out"+str(t)+".dae", str(t))

			print("[sample interpolation]")

	def test(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		# Get 16 models using IM-AE.
		for t in range(16):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						batch_voxels = self.data_voxels[t:t+1]
						model_out = self.sess.run(self.sG,
							feed_dict={
								self.vox3d: batch_voxels,
								self.point_coord: self.coords[minib],
							})
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

			thres = 0.5
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+"out"+str(t)+".dae", str(t))

			print("[sample]")

	def get_z(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		hdf5_path = self.data_dir+'/'+self.dataset_name+'_z.hdf5'
		chair_num = len(self.data_voxels)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		# Create a dataset with shape [chair_num, 128]
		hdf5_file.create_dataset("zs", [chair_num,self.z_dim], np.float32)

		for idx in range(0, chair_num):
			print(idx)
			batch_voxels = self.data_voxels[idx:idx+1]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			hdf5_file["zs"][idx,:] = np.reshape(z_out,[self.z_dim])

		print(hdf5_file["zs"].shape)
		hdf5_file.close()
		print("[z]")

	def test_z(self, config, batch_z, dim):
		# Use a batch of z to recover the shape.
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier

		#get coords 256
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])

		for t in range(batch_z.shape[0]):
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						print(t,i,j,k)
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

			thres = 0.5
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+"out"+str(t)+".dae", str(t))

			print("[sample GAN]")

	def test_z_pc(self, config, batch_z, dim):
		# Point cloud representation.
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		dima = self.test_size #32
		multiplier = int(dim/dima) # 2
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier

		#get coords 256
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32) # 8*32*32*32*3
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])

		# Choose 2048 points from the model.
		n_pc_points = 2048
		thres = 0.5
		hdf5_file = h5py.File(self.dataset_name + "_im_gan_sample.hdf5", 'w')
		# Points only have x, y, z information.
		hdf5_file.create_dataset("points", [batch_z.shape[0],n_pc_points,3], np.float32)

		for t in range(batch_z.shape[0]):
			print(t)
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])

			# Extract marching mesh from the 66*66*66 cube.
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+"out"+str(t)+".dae", str(t))
			np.random.shuffle(vertices)
			# ???
			vertices = (vertices - dim/2 - 0.5)/dim

			vertices_out = np.zeros([n_pc_points,3], np.float32)
			vertices_len = vertices.shape[0]
			for i in range(n_pc_points):
				# If the number is less than 2048, then start from the first point and continue.
				vertices_out[i] = vertices[i%vertices_len]

			hdf5_file["points"][t,:,:] = vertices_out

		hdf5_file.close()

	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.input_size)

	def save(self, checkpoint_dir, step):
		model_name = "IMAE.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

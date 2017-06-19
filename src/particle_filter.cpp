/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <iostream>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <limits>
#include "particle_filter.h"


const double ParticleFilter::EPSILON_CTE = 0.0001;
static std::mt19937 mt_rnd(1234); // Use Mersenne twister.

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = ParticleFilter::NUMPARTICLES; 

	std::normal_distribution<double> rnd_x(0, std[0]); // Add the uncertainties as a Gauissian distribution.
	std::normal_distribution<double> rnd_y(0, std[1]); 
	std::normal_distribution<double> rnd_theta(0, std[2]); 

	for (size_t i = 0; i < (size_t)num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = x + rnd_x(mt_rnd);
		p.y = y + rnd_y(mt_rnd);
		p.theta = theta + rnd_theta(mt_rnd);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::normal_distribution<double> rnd_x(0, std_pos[0]);
	std::normal_distribution<double> rnd_y(0, std_pos[1]); 
	std::normal_distribution<double> rnd_theta(0, std_pos[2]); 

	for (size_t i = 0; i < (size_t)num_particles; ++i) {
		particles[i].x += rnd_x(mt_rnd);
		particles[i].y += rnd_y(mt_rnd);
		particles[i].theta += rnd_theta(mt_rnd);
		if (fabs(yaw_rate) < ParticleFilter::EPSILON_CTE) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * 
				(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * 
				(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double min_distance;
	int id;
	for (size_t i = 0; i < observations.size(); ++i) {
		min_distance = (double)std::numeric_limits<int>::max();
		id = std::numeric_limits<int>::min();
		for (size_t j = 0; j < predicted.size(); ++j) {
			// Scan the predicted landmarks and match these with the closest observation.
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_distance) {
				min_distance = distance;
				id = predicted[j].id;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	double diff_x, diff_y, t_x, t_y;
	bool found;
	size_t index;

	for (size_t i = 0; i < (size_t)num_particles; ++i) {
		std::vector<LandmarkObs> predictions;
		for (size_t j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			diff_x = (double)map_landmarks.landmark_list[j].x_f - particles[i].x;
			diff_y = (double)map_landmarks.landmark_list[j].y_f - particles[i].y;
			if (diff_x * diff_x + diff_y * diff_y <= sensor_range * sensor_range) { // It is in range: include the point.
				predictions.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,  
					(double)map_landmarks.landmark_list[j].x_f, 
					(double)map_landmarks.landmark_list[j].y_f});
			}
		}
		// Now run the transformation following http://planning.cs.uiuc.edu/node99.html.
		std::vector<LandmarkObs> transformations;
		for (size_t j = 0; j < observations.size(); ++j) {
			t_x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * 
				observations[j].y + particles[i].x;
			t_y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * 
				observations[j].y + particles[i].y;
			transformations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
		}
		
		dataAssociation(predictions, transformations);
		
		particles[i].weight = 1.0;
		for (size_t j = 0; j < transformations.size(); ++j) {
			found = false; 
			index = 0;
			while (!found && index < predictions.size()) { // Seek the index.
				if (predictions[index].id == transformations[j].id) {
					found = true;
				} else {
					++index;
				}
			}
			// Compute the weight.
			particles[i].weight *= (1.0 /(2.0 * M_PI * std_landmark[0] * std_landmark[1])) * 
				exp(-(pow(predictions[index].x - transformations[j].x, 2.0) / (2.0 * std_landmark[0] * std_landmark[0]) + 
				(pow(predictions[index].y - transformations[j].y, 2) / (2 * std_landmark[1] * std_landmark[1]))));
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> new_particles;
	std::vector<double> weights;
	size_t w_index = 0;
	double max_weight = (double)std::numeric_limits<int>::min();
	for (size_t i = 0; i < particles.size(); ++i) {
		weights.push_back(particles[i].weight);
		if (max_weight < particles[i].weight) {
			max_weight = particles[i].weight;
		}
	}

	double beta = 0.0;
	std::discrete_distribution<unsigned int> rndg(weights.begin(), weights.end());
	std::uniform_real_distribution<double> beta_rnd(0.0, max_weight);
	w_index = rndg(mt_rnd);
	for (size_t i = 0; i < num_particles; ++i) {
		beta = 2.0 * beta_rnd(mt_rnd);
		while (beta > weights[w_index]) {
			beta -= weights[w_index];
			w_index = (w_index + 1) % num_particles; // We avoid negative indices this way.
		}
		new_particles.push_back(particles[w_index]);
	}
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

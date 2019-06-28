import numpy as np

class NeuralNetwork():
    
    def __init__(self, inputs):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        #this range ensures that the gradients will not be too low and prevent learning
        self.synaptic_weights = 2 * np.random.random((inputs, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        #converts integers to percentages to work with so they lie between 0 and 1
        inputs = inputs.astype(float) / 100
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork(2)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 5 examples--2 input values and 1 output 
    #[Hours Studied, Midterm Test Score] and [Pass or Fail] Respecively
    training_inputs = np.array([[35, 67],
                                [12, 75],
                                [16, 89],
                                [45, 56],
                                [10, 90]])

    training_outputs = np.array([[1,0,1,1,0]]).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    #user input in terminal
    hours_studied = str(input("Hours Studied: "))
    midterm_score = str(input("Midterm Score: "))
        
    print("Considering New Situation: ", hours_studied, midterm_score)
    print("Pass or Fail Final: ")
    
    outcome = neural_network.think(np.array([hours_studied, midterm_score]))
    if outcome > 0.98:
        print("Pass at: ")
    else:
        print("Fail at: ")
    
    print(outcome)

/*

+===========================================================+
|  _____ _             _   ____            _           _    |
| |  ___(_)_ __   __ _| | |  _ \ _ __ ___ (_) ___  ___| |_  |
| | |_  | | '_ \ / _` | | | |_) | '__/ _ \| |/ _ \/ __| __| |
| |  _| | | | | | (_| | | |  __/| | | (_) | |  __/ (__| |_  |
| |_|   |_|_| |_|\__,_|_| |_|   |_|  \___// |\___|\___|\__| |
|                                       |__/                |
+===========================================================+

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
| A SIMPLE ARTIFICIAL NEURAL NETWORK WRITTEN IN C++ |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

+--------------------------+
| Liam Mulhall             |
| Instructor: Shayon Gupta |
| TA: Alexander Curtiss    |
| Data Structures          |
| Fall 2018                |
+--------------------------+

+----------------------------------------------------------------+
| "A Simple Artificial Neural Network Written in C++" by Liam    |
| Mulhall is licensed under CC BY-NC-SA 3.0 US.                  |
|                                                                |
| Link to License:                                               |
| https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| Much of the code for the neural network in "A Simple           |
| Artificial Neural Network Written in C++" by Liam Mulhall is   |
| adapted from David Miller's tutorial.                          |
|                                                                |
| Link to Tutorial:                                              |
| https://vimeo.com/19569529                                     |
|                                                                |
| "Neural Net in C++ Tutorial" by David Miller is licensed under |
| CC BY-NC-SA 3.0 US.                                            |
|                                                                |
| Link to License:                                               |
| https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode |
|                                                                |
| The Python code is adapted from Sentdex's tutorial.            |
|                                                                |
| Link to Tutorial:                                              |
| https://youtu.be/QyhqzaMiFxk                                   |
+----------------------------------------------------------------+

+----------------------------------------------------------+
| The main function of this program is written so that the |
| neural network can "learn" to behave like an XOR gate.   |
|                                                          |
| To change the behavior of the neural network, change the |
| main function and use appropriate data.                  |
|                                                          |
| This program was written and tested on macOS.            |
+----------------------------------------------------------+

+--------------------------------+
| ======== DEPENDENCIES ======== |
| (1) gxx 4.2.1                  |
| (2) Python 3.7.1               |
| (3) Matplotlib 3.0.2           |
| (4) NumPy 1.15.4               |
+--------------------------------+

+--------------------------------+
| ======== INSTRUCTIONS ======== |
| (1) Download "Error-Graph.py"  |
|     and move it to the same    |
|     directory in which this    |
|     program resides.           |
| (2) Compile using g++.         |
| (3) Execute.                   |
|                                |
| - No command line arguments    |
|   required.                    |
| - No data files required.      |
+--------------------------------+

*/

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

//////////////////////////////////////////////////
// MISCELLANEA NEEDED FOR NETWORK AND NEURONS
//////////////////////////////////////////////////

// Forward declaration of "Neuron" class.
class Neuron;

// For weighted connections between neurons.
struct Connection {
  double weight;
  double deltaWeight;
};

// A new type for simplicity's sake.
typedef vector<Neuron> Layer;

//////////////////////////////////////////////////
// "Neuron" CLASS DECLARATIONS
//////////////////////////////////////////////////

class Neuron {
  public:
    // Constructor
    Neuron(int numOutputs, int desiredIndex);
    // PubF#1
    void calcHiddenGradients(Layer &nextLayer);
    // PubF#2
    void calcOutputGradients(double targetValue);
    // PubF#3
    void feedForward(Layer &prevLayer);
    // PubF#4
    double getOutputValue();
    // PubF#5
    void setOutputValue(double desiredValue);
    // PubF#6
    void updateInputWeights(Layer &prevLayer);
  private:
    // PriF#1
    double randomWeight();
    // PriF#2
    double sumDOW(Layer &nextLayer);
    // PriF#3
    double transferFunction(int n, double x);
    // PriF#4
    double transferFunctionDerivative(int n, double x);
    // V#1
    static double alpha; // Momentum.
    // V#2
    static double eta; // Training rate.
    // V#3
    double gradient;
    // V#4
    int index;
    // V#5
    double outputValue;
    // V#6
    vector<Connection> outputWeights;
};

//////////////////////////////////////////////////
// "Neuron" CLASS IMPLEMENTATIONS
//////////////////////////////////////////////////

// Constructor
Neuron::Neuron(int numOutputs, int desiredIndex) {
  for(int i = 0; i < numOutputs; i++) {
    outputWeights.push_back(Connection());
    outputWeights.back().weight = randomWeight();
  }
  index = desiredIndex;
}

// PubF#1
void Neuron::calcHiddenGradients(Layer &nextLayer) {
  double dow = sumDOW(nextLayer);
  gradient = dow * transferFunctionDerivative(1, outputValue);
}

// PubF#2
void Neuron::calcOutputGradients(double targetValue) {
  double delta = targetValue - outputValue;
  gradient = delta * transferFunctionDerivative(1, outputValue);
}

// PubF#3
void Neuron::feedForward(Layer &prevLayer) {

  // Sum previous layer's outputs multiplied by connection weights.
  double sum = 0.0;
  for(int i = 0; i < prevLayer.size(); i++) {
    sum += prevLayer[i].getOutputValue()
        * prevLayer[i].outputWeights[index].weight;
  }
  outputValue = transferFunction(1, sum);
}

// PubF#4
double Neuron::getOutputValue() {
  return outputValue;
}

// PubF#5
void Neuron::setOutputValue(double desiredValue) {
  outputValue = desiredValue;
}

// PubF#6
void Neuron::updateInputWeights(Layer &prevLayer) {
  for(int i = 0; i < prevLayer.size(); i++) {
    Neuron &neuron = prevLayer[i];
    double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
    double newDeltaWeight = eta * neuron.getOutputValue() * gradient + alpha
                          * oldDeltaWeight;
    neuron.outputWeights[index].deltaWeight = newDeltaWeight;
    neuron.outputWeights[index].weight += newDeltaWeight;
  }
}

// PriF#1
double Neuron::randomWeight() {
  return rand() / double(RAND_MAX);
}

// PriF#2
double Neuron::sumDOW(Layer &nextLayer) {
  double sum = 0.0;
  for(int i = 0; i < nextLayer.size() - 1; i++) {
    sum += outputWeights[i].weight * nextLayer[i].gradient;
  }
  return sum;
}

// PriF#3
double Neuron::transferFunction(int n, double x) {
  switch(n) {

    // Hyperbolic tangent function.
    case 1:
      return tanh(x);
      break;

    // Inverse tangent function.
    case 2:
      return atan(x);

    // Softsign function.
    case 3:
      return x / (1 + abs(x));
      break;

    // Square nonlinearity function.
    case 4:
      if(x > 2.0) {
        return 1;
      }
      else if(0 <= x && x <= 2.0) {
        return x - pow(x, 2) / 4;
      }
      else if(-2.0 <= x && x < 0) {
        return x + pow(x, 2) / 4;
      }
      else {
        return -1;
      }
      break;
  }
  cout << endl;
  cout << "Error: Wrong integer passed to 'transferFunction'." << endl;
  cout << endl;
  return INT_MIN;
}

// PriF#4
double Neuron::transferFunctionDerivative(int n, double x) {
  switch(n) {

    // Hyperbolic tangent function derivative approximation.
    case 1:
      return 1.0 - pow(x, 2);
      break;

    // Inverse tangent function derivative.
    case 2:
      return 1.0 / (1.0 + pow(x, 2));
      break;

    // Softsign function derivative.
    case 3:
      return 1.0 / (pow((1.0 + abs(x)), 2));
      break;

    // Square nonlinearity function derivative.
    case 4:
      if(x > 2.0) {
        return 0;
      }
      else if(0 <= x && x <= 2.0) {
        return 1 - x / 2;
      }
      else if(-2.0 <= x && x < 0) {
        return 1 + x / 2;
      }
      else {
        return 0;
      }
      break;
  }
  cout << endl;
  cout << "Error: Wrong integer passed to 'transferFunctionDerivative'."
       << endl;
  cout << endl;
  return INT_MIN;
}

// V#1
double Neuron::alpha = 0.5; // Momentum.

// V#2
double Neuron::eta = 0.15; // Training rate.

//////////////////////////////////////////////////
// "Network" CLASS DECLARATIONS
//////////////////////////////////////////////////

class Network {
  public:
    // Constructor
    Network(vector<int> &topology);
    // PubF#1
    void backPropogate(vector<double> &targetValues);
    // PubF#2
    void feedForward(vector<double> &inputValues);
    // PubF#3
    double getRecentAverageError();
    // PubF#4
    void getResults(vector<double> &resultValues);
  private:
    // V#1
    double error;
    // V#2
    vector<Layer> layers;
    // V#3
    double recentAverageError;
    // V#4
    static double recentAverageSmoothingFactor;
};

//////////////////////////////////////////////////
// "Network" CLASS IMPLEMENTATIONS
//////////////////////////////////////////////////

// Constructor
Network::Network(vector<int> &topology) {

  // Add layers to neural network.
  for(int i = 0; i < topology.size(); i++) {
    layers.push_back(Layer());
    cout << "Layer " << i + 1 << " made!" << endl;
    int numOutputs;
    if(i == topology.size() - 1) {
      numOutputs = 0;
    }
    else {
      numOutputs = topology[i + 1];
    }

    // Add neurons to layer. Also add bias neurons to layer, hence "<=".
    for(int j = 0; j <= topology[i]; j++){
      layers.back().push_back(Neuron(numOutputs, j));
      if(j < topology[i]) {
        cout << "  - Made neuron " << j + 1 << " in layer " << i + 1 << "!"
             << endl;
      }
      else {
        cout << "  - Made bias neuron in layer " << i + 1 << "!"
             << endl;
      }
    }

    // Make output of bias neurons 1.0.
    layers.back().back().setOutputValue(1.0);
  }
}

// PubF#1
void Network::backPropogate(vector<double> &targetValues) {

  // Calculate overall root-mean-square error.
  Layer &outputLayer = layers.back();
  error = 0.0;
  for(int i = 0; i < outputLayer.size() - 1; i++) {
    double delta = targetValues[i] - outputLayer[i].getOutputValue();
    error += pow(delta, 2);
  }
  error /= outputLayer.size() - 1;
  error = sqrt(error);

  // Set "recentAverageError".
  recentAverageError = (recentAverageError * recentAverageSmoothingFactor
                     + error) / (recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients.
  for(int i = 0; i < outputLayer.size() - 1; i++) {
    outputLayer[i].calcOutputGradients(targetValues[i]);
  }

  // Calculate hidden layer gradients.
  for(int i = layers.size() - 2; i > 0; i--) {
    Layer &hiddenLayer = layers[i];
    Layer &nextLayer = layers[i + 1];

    // Iterate through neurons.
    for(int j = 0; j < hiddenLayer.size(); j++) {
      hiddenLayer[j].calcHiddenGradients(nextLayer);
    }
  }

  // Update connection weights.
  for(int i = layers.size() - 1; i > 0; i--) {
    Layer &currentLayer = layers[i];
    Layer &prevLayer = layers[i - 1];

    // Iterate through neurons.
    for(int j = 0; j < currentLayer.size() - 1; j++) {
      currentLayer[j].updateInputWeights(prevLayer);
    }
  }
}

// PubF#2
void Network::feedForward(vector<double> &inputValues) {

  // Add input values to neural network.
  for(int i = 0; i < inputValues.size(); i++) {
    layers[0][i].setOutputValue(inputValues[i]);
  }

  // Forward propogate.
  for(int i = 1; i < layers.size(); i++) {
    Layer &prevLayer = layers[i - 1];

    // Iterate through neurons.
    for(int j = 0; j < layers[i].size() - 1; j++) {
      layers[i][j].feedForward(prevLayer);
    }
  }
}

// PubF#3
double Network::getRecentAverageError() {
  return recentAverageError;
}

// PubF#4
void Network::getResults(vector<double> &resultValues) {
  resultValues.clear();
  for(int i = 0; i < layers.back().size() - 1; i++) {
    resultValues.push_back(layers.back()[i].getOutputValue());
  }
}

// V#4
double Network::recentAverageSmoothingFactor = 100.0;

//////////////////////////////////////////////////
// MISCELLANEA NEEDED FOR CLI
//////////////////////////////////////////////////

void displayBuildQuestion() {
  cout << "Would you like to build a neural network?" << endl;
  cout << "  - Enter 'y' to build a neural network." << endl;
  cout << "  - Enter 'n' to quit." << endl;
  cout << "Answer: ";
}

void displayBuilt() {
  cout << "+---------------------------------+" << endl;
  cout << "| The neural network build was a  |" << endl;
  cout << "| success!                        |" << endl;
  cout << "+---------------------------------+" << endl;
}

void displayCompletion() {
  cout << "+---------------------------------+" << endl;
  cout << "| Training complete. The results  |" << endl;
  cout << "| were written to 'Results.txt',  |" << endl;
  cout << "| which is located in the primary |" << endl;
  cout << "| working directory.              |" << endl;
  cout << "|                                 |" << endl;
  cout << "| A graph of the recent average   |" << endl;
  cout << "| error versus the training round |" << endl;
  cout << "| can also be found in the        |" << endl;
  cout << "| primary working directory.      |" << endl;
  cout << "+---------------------------------+" << endl;
}

void displayInfo() {
  cout << "+---------------------------------+" << endl;
  cout << "| This is a program for a simple  |" << endl;
  cout << "| artificial neural network.      |" << endl;
  cout << "|                                 |" << endl;
  cout << "| A neural network is essentially |" << endl;
  cout << "| a graph with layers (think      |" << endl;
  cout << "| columns) of nodes called        |" << endl;
  cout << "| neurons. Each neuron has a      |" << endl;
  cout << "| weighted connection to each     |" << endl;
  cout << "| neuron in the layer to the      |" << endl;
  cout << "| right.                          |" << endl;
  cout << "|                                 |" << endl;
  cout << "| Our objective will be to build  |" << endl;
  cout << "| a neural network and train it   |" << endl;
  cout << "| to behave like an XOR gate.     |" << endl;
  cout << "+---------------------------------+" << endl;
}

void displayTestAfterQuestion() {
  cout << "Would you like to test the neural network now that it has been ";
  cout << "trained?" << endl;
  cout << "  - Enter 'y' to test the neural network." << endl;
  cout << "  - Enter 'n' to quit."
       << endl;
  cout << "Answer: ";
}

void displayTestBeforeQuestion() {
  cout << "Would you like to test the neural network before training it?"
       << endl;
  cout << "  - Enter 'y' to test the neural network." << endl;
  cout << "  - Enter 'n' to continue without testing the neural network."
       << endl;
  cout << "Answer: ";
}

void displayTrainQuestion() {
  cout << "Would you like to train the neural network?" << endl;
  cout << "  - Enter 'y' to train the neural network." << endl;
  cout << "  - Enter 'n' to quit." << endl;
  cout << "Answer: ";
}

void displayTrainingRoundsQuestion() {
  cout << "How many training rounds would you like to perform?" << endl;
  cout << "  - Please enter a number between 1000 and 100000." << endl;
  cout << "Enter number of training rounds: ";
}

void displayWelcome() {
  cout << "+---------------------------------+" << endl;
  cout << "|             WELCOME             |" << endl;
  cout << "+---------------------------------+" << endl;
}

void displayXOR() {
  cout << "+---------------------------------+" << endl;
  cout << "| An XOR (exclusive or) gate is a |" << endl;
  cout << "| logic gate that accepts two     |" << endl;
  cout << "| inputs and produces one output. |" << endl;
  cout << "|                                 |" << endl;
  cout << "| A logic gate produces true (1)  |" << endl;
  cout << "| when there is only one true     |" << endl;
  cout << "| input and produces false (0)    |" << endl;
  cout << "| when there are two inputs of    |" << endl;
  cout << "| the same kind.                  |" << endl;
  cout << "|                                 |" << endl;
  cout << "| ======= XOR Truth Table ======= |" << endl;
  cout << "|       +-------+---------+       |" << endl;
  cout << "|       | INPUT | OUTPUT  |       |" << endl;
  cout << "|       +-------+---------+       |" << endl;
  cout << "|       | A | B | A XOR B |       |" << endl;
  cout << "|       +-------+---------+       |" << endl;
  cout << "|       | 0 | 0 |    0    |       |" << endl;
  cout << "|       | 0 | 1 |    1    |       |" << endl;
  cout << "|       | 1 | 0 |    1    |       |" << endl;
  cout << "|       | 1 | 1 |    0    |       |" << endl;
  cout << "|       +-------+---------+       |" << endl;
  cout << "+---------------------------------+" << endl;
}

//////////////////////////////////////////////////
// MISCELLANEA NEEDED FOR FILES
//////////////////////////////////////////////////

void callPython() {
  string command = "python3 Error-Graph.py";
  system(command.c_str());
}

double getNum() {
  return rand() % 2;
}

double getTarget(double input1, double input2) {
  if(input1 == input2) {
    return 0.0;
  }
  else {
    return 1.0;
  }
}

void makeTrainingFile(int desiredTrainingRounds) {
  ofstream ofs;
  ofs.open("Training.csv");
  double input1, input2, target;
  for(int i = 0; i < desiredTrainingRounds; i++) {
     input1 = getNum();
     input2 = getNum();
     target = getTarget(input1, input2);
     ofs << input1 << "," << input2 << "," << target << endl;
  }
}

//////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////

int main() {

  // Greet and inform user.
  cout << endl;
  displayWelcome();
  cout << endl;
  displayInfo();
  cout << endl;
  displayXOR();
  cout << endl;

  // Ask user if they want to build a neural network.
  char buildAnswer;
  displayBuildQuestion();
  cin >> buildAnswer;
  if(buildAnswer == 'n'){
    cout << endl;
    cout << "Goodbye!" << endl;
    cout << endl;
    return 0;
  }
  cout << endl;

  // Prompt user for number of layers.
  int numLayers;
  cout << "Enter number of layers: ";
  cin >> numLayers;

  // Prompt user for number of neurons.
  int numNeurons;
  vector<int> topology;
  for(int i = 0; i < numLayers; i++) {
    cout << "  - Enter number of neurons for layer " << i + 1 << ": ";
    cin >> numNeurons;
    topology.push_back(numNeurons);
  }
  cout << endl;

  // Build neural network.
  Network N(topology);
  cout << endl;
  displayBuilt();
  cout << endl;

  // Miscellanea needed for test.
  string userInput1;
  string userInput2;
  vector<double> inputValues, resultValues;

  // Ask user if they want to test neural network before training.
  char testBeforeAnswer;
  displayTestBeforeQuestion();
  cin >> testBeforeAnswer;
  if(testBeforeAnswer == 'y') {

    // Prompt user for inputs.
    cout << endl;
    cout << "========== TEST ==========" << endl;
    cout << "Enter input 1: ";
    cin >> userInput1;
    inputValues.push_back(stod(userInput1));
    cout << "Enter input 2: ";
    cin >> userInput2;
    inputValues.push_back(stod(userInput2));

    // Feed forward input values and clear input values.
    N.feedForward(inputValues);
    inputValues.clear();

    // Get results.
    N.getResults(resultValues);

    // Print target values.
    cout << "Target: " << getTarget(stod(userInput1), stod(userInput2))
         << endl;

    // Print result values and clear result values.
    cout << "Result: " << resultValues[0] << endl;
    resultValues.clear();
    cout << "==========================" << endl;
    cout << endl;
  }
  cout << endl;

  // Ask user if they want to train neural network.
  char trainAnswer;
  displayTrainQuestion();
  cin >> trainAnswer;
  if(trainAnswer == 'n'){
    cout << endl;
    cout << "Goodbye!" << endl;
    cout << endl;
    return 0;
  }
  cout << endl;

  // Ask user for number of training rounds.
  int desiredTrainingRounds;
  displayTrainingRoundsQuestion();
  cin >> desiredTrainingRounds;
  cout << endl;

  // Miscellanea needed for reading file.
  int trainingRound = 1;
  string line;
  string item;
  vector<double> targetValues;

  // Make and open files.
  ofstream errorFile;
  errorFile.open("Error.csv");
  ofstream resultsFile;
  resultsFile.open("Results.txt");
  makeTrainingFile(desiredTrainingRounds);
  ifstream trainingFile;
  trainingFile.open("Training.csv");
  if(errorFile.is_open() && resultsFile.is_open() && trainingFile.is_open()) {

    // Parse each line.
    while(getline(trainingFile, line)) {
      resultsFile << "==== Training Round " << trainingRound << " ===="
                  << endl;
      istringstream iss(line);
      getline(iss, item, ',');
      inputValues.push_back(stod(item));
      getline(iss, item, ',');
      inputValues.push_back(stod(item));
      getline(iss, item);
      targetValues.push_back(stod(item));

      // Write input values to results file.
      for(int i = 0; i < inputValues.size(); i++) {
        resultsFile << "Input " << i + 1 << ": " << inputValues[i] << endl;
      }

      // Feed forward input values and clear input values.
      N.feedForward(inputValues);
      inputValues.clear();

      // Get results.
      N.getResults(resultValues);

      // Write target and result values to results file.
      resultsFile << "Target: " << targetValues[0] << endl;
      resultsFile << "Result: " << resultValues[0] << endl;
      resultValues.clear();

      // Correct weights.
      N.backPropogate(targetValues);
      targetValues.clear();

      // Write error statistics to results file.
      resultsFile << "Recent Average Error = " << N.getRecentAverageError()
                  << endl;
      resultsFile << "==========================" << endl;
      resultsFile << endl;

      // Add values to error file.
      errorFile << trainingRound << "," << N.getRecentAverageError() << endl;
      trainingRound++;
    }
    errorFile.close();
    resultsFile.close();
    trainingFile.close();
  }
  callPython();
  displayCompletion();
  cout << endl;

  // Ask user if they want to test neural network after having trained it.
  char testAfterAnswer = 'a';
  while(testAfterAnswer != 'n') {
    displayTestAfterQuestion();
    cin >> testAfterAnswer;
    if(testAfterAnswer == 'n') {
      cout << endl;
      cout << "Goodbye!" << endl;
      cout << endl;
      break;
    }

    // Prompt user for inputs.
    cout << endl;
    cout << "========== TEST ==========" << endl;
    cout << "Enter input 1: ";
    cin >> userInput1;
    inputValues.push_back(stod(userInput1));
    cout << "Enter input 2: ";
    cin >> userInput2;
    inputValues.push_back(stod(userInput2));

    // Feed forward input values and clear input values.
    N.feedForward(inputValues);
    inputValues.clear();

    // Get results.
    N.getResults(resultValues);

    // Print target values.
    cout << "Target: " << getTarget(stod(userInput1), stod(userInput2))
         << endl;

    // Print result values and clear result values.
    cout << "Result: " << resultValues[0] << endl;
    resultValues.clear();
    cout << "==========================" << endl;
    cout << endl;
  }

  return 0;
}

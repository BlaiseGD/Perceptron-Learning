#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <cmath>

/*

GOALS, NOTES, REMINDERS

GOALS:

big goal is to make a linear neural network and then after I can make a non-linear neural network for something else. When I make a non-linear Neural Net it, will require a complex activation function to capture non-linearity in output

Answer key function (Done)
Random coord gen within domain (Done)
PLR (perceptron learning rule) / Training Method (Done)
Easy Perceptron Instantiation (Done)

NOTES:

the PLR method only changes the weights and biases for the next generation and this depends on the result of the activation function (e.g. 0 or 1)
PLR Logic is to change weights according to this formula W[i] = W[i] + LearningRate * error * input[i], Bias is B[i] = B[i] + learning rate * error


REMINDERS:
post to Github for safekeeping and to practice using git from the cli


*/

const double X_MAX = 300;
const double Y_MAX = 153; // this is the output when given the maximum x value intook by the function
const double LEARNING_RATE = 0.01;
double bias = 1.000;
std::random_device rd;
std::mt19937 gen(rd());

double randZeroToOne()
{
    // THIS FUNCTION IS ALSO THE X-COORD GENERATOR WHEN MULTIPLIED BY 100
    std::uniform_int_distribution<> dist(0, 100);
    return dist(gen) / 100.0;
    // did this weird div by 100 because it makes it only have two decimal places bc we all know computers are bad at decimal math. Gotta limit the error
    //two decimal places so it's an integer when multiplied by 300 later
}

double linearFunction(double x) { return 0.5 * x + 3; }

double generateX() { return randZeroToOne() * X_MAX; }
double generateY() { return randZeroToOne() * Y_MAX; }

class Perceptron
{
public:
    double inputNumber;
    const double bias = 1.00;
    double biasWeight = randZeroToOne();
    double weightX = randZeroToOne();
    double weightY = randZeroToOne();
    std::vector<std::pair<double, double>> inputs;

    Perceptron() {}

    int activationFunction(double x, double y) const; // chatGPT told me to put const at the end of this so it would work for the printing function it wrote so I obliged
    void trainingFunction(std::vector<std::pair<double, double>> inputs);
    void fillInputs();
    void predict();
};

int label(double x, double y)
{
    return y > linearFunction(x) ? 1 : 0;
}

void Perceptron::fillInputs()
{
    for (int i = 0; i < inputNumber; i++)
    {
        inputs.push_back({ generateX(), generateY() });
    }
}

void Perceptron::trainingFunction(std::vector<std::pair<double, double>> inputs)
{
    const int maxEpochs = 10000;
    const int minEpochs = 5;
    int epochs = 0;

    while (epochs < maxEpochs)
    {
        int totalErrors = 0;
        for (auto& i : inputs)
        {
            int error = label(i.first, i.second) - activationFunction(i.first, i.second);
            // label is target and activationfunction is predicted
            if (error != 0)
            {
                weightX += error * LEARNING_RATE * (i.first / X_MAX);
                weightY += error * LEARNING_RATE * (i.second / Y_MAX);
                biasWeight += LEARNING_RATE * error * bias;
                totalErrors++;
            }
        }
        epochs++;
        std::cout << "total Errors: " << totalErrors << std::endl;
        if (totalErrors == 0)
        {
            std::cout << "epoch #: " << epochs << std::endl;
            break;
        }
    }
}

int Perceptron::activationFunction(double x, double y) const
{
    // This will be the implementation of the Perceptron Learning Rule
    double sum = weightX * (x / X_MAX) + weightY * (y / Y_MAX) + bias * biasWeight;
    return (sum > 0) ? 1 : 0;
}

// print function courtesy of ChatGPT because I was not writing a print function after writing all the other code
void printResults(const Perceptron& p)
{
    std::cout << "\nPerceptron results vs actual:\n";
    for (const auto& point : p.inputs)
    {
        int predicted = p.activationFunction(point.first, point.second);
        int actual = label(point.first, point.second);
        std::cout << "Point (" << point.first << ", " << point.second << "): "
            << "Predicted: " << predicted
            << ", Actual: " << actual
            << (predicted == actual ? " [Correct]" : " [Incorrect]")
            << std::endl;
    }
}

int main()
{
    Perceptron theChosenOne;

    std::cout << "How many inputs do you want to train with? ";
    std::cin >> theChosenOne.inputNumber;
    theChosenOne.fillInputs();
    theChosenOne.trainingFunction(theChosenOne.inputs);
    printResults(theChosenOne);

    return 0;
}

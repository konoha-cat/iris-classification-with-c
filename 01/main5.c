#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define INPUTSIZE 5  // Including bias
#define HIDDENSIZE 5  // Including bias
#define OUTPUTSIZE 3
#define K 5

#define LOOPNUM 500

typedef struct {
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
    int species;
} Iris;

double sigmoid(double x);
void apply_softmax(double *input, double *output, int size);
void Forward(double *input, double wih[][INPUTSIZE], double who[][HIDDENSIZE], double *bias_hidden, double *bias_output, double *hidden, double *output);
void Backward(double *input, double *hidden, double *output, double *teacher, double wih[][INPUTSIZE], double who[][HIDDENSIZE], double *bias_hidden, double *bias_output, double eta);
void k_fold_cross_validation(Iris iris_data[], int data_size);
double train_and_test(Iris iris_data[], int data_size, const char* operation);

int read(Iris iris_data[]) {
    FILE *file = fopen("iris.csv", "r");
    if (!file) {
        printf("Error with opening file.\n");
        return -1;
    }

    const int max_line_length = 100;
    char line[max_line_length];
    int count = 0;

    // Skip first row
    fgets(line, max_line_length, file);

    // Read data cols
    while (fgets(line, max_line_length, file)) {
        char species_str[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s",
               &iris_data[count].sepal_length,
               &iris_data[count].sepal_width,
               &iris_data[count].petal_length,
               &iris_data[count].petal_width,
               species_str);

        if (strcmp(species_str, "'setosa'") == 0) {
            iris_data[count].species = 0;
        } else if (strcmp(species_str, "'versicolor'") == 0) {
            iris_data[count].species = 1;
        } else if (strcmp(species_str, "'virginica'") == 0) {
            iris_data[count].species = 2;
        } else {
            printf("Unknown: %s\n", species_str);
        }

        count++;
    }

    fclose(file);

    return count;
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void apply_softmax(double *input, double *output, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += exp(input[i] - max);
    }

    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max) / sum;
    }
}

void Forward(double *input, double wih[][INPUTSIZE], double who[][HIDDENSIZE], double *bias_hidden, double *bias_output, double *hidden, double *output) {
    // Add bias neuron to input
    input[INPUTSIZE-1] = 1.0;

    // Hidden layer computations
    for (int j = 0; j < HIDDENSIZE-1; j++) {
        hidden[j] = 0.0;
        for (int i = 0; i < INPUTSIZE; i++) {
            hidden[j] += wih[j][i] * input[i];
        }
        hidden[j] = sigmoid(hidden[j]);
    }
    hidden[HIDDENSIZE-1] = 1.0;  // Bias neuron for hidden layer

    // Output layer computations
    double pre_softmax[OUTPUTSIZE];
    for (int j = 0; j < OUTPUTSIZE; j++) {
        pre_softmax[j] = 0.0;
        for (int i = 0; i < HIDDENSIZE; i++) {
            pre_softmax[j] += who[j][i] * hidden[i];
        }
    }

    // Apply softmax to output layer
    apply_softmax(pre_softmax, output, OUTPUTSIZE);
}


void Backward(double *input, double *hidden, double *output, double *teacher, double wih[][INPUTSIZE], double who[][HIDDENSIZE], double *bias_hidden, double *bias_output, double eta) {
    double delta_output[OUTPUTSIZE];
    double delta_hidden[HIDDENSIZE];

    // Compute output layer deltas
    for (int j = 0; j < OUTPUTSIZE; j++) {
        delta_output[j] = teacher[j] - output[j];
    }

    // Compute hidden layer deltas
    for (int j = 0; j < HIDDENSIZE-1; j++) {
        double error = 0.0;
        for (int k = 0; k < OUTPUTSIZE; k++) {
            error += delta_output[k] * who[k][j];
        }
        delta_hidden[j] = error * hidden[j] * (1.0 - hidden[j]);
    }
    

    // Update output layer weights
    for (int j = 0; j < OUTPUTSIZE; j++) {
        for (int i = 0; i < HIDDENSIZE-1; i++) {
            who[j][i] += eta * delta_output[j] * hidden[i];
        }
        bias_output[j] += eta * delta_output[j];
    }

    // Update hidden layer weights
    for (int j = 0; j < HIDDENSIZE-1; j++) {
        for (int i = 0; i < INPUTSIZE-1; i++) {
            wih[j][i] += eta * delta_hidden[j] * input[i];
        }
        bias_hidden[j] += eta * delta_hidden[j];
    }
}

double train_and_test(Iris iris_data[], int data_size, const char* operation) {
    double input[INPUTSIZE], hidden[HIDDENSIZE], output[OUTPUTSIZE];
    double weight_ih[HIDDENSIZE][INPUTSIZE], weight_ho[OUTPUTSIZE][HIDDENSIZE];
    double bias_hidden[HIDDENSIZE], bias_output[OUTPUTSIZE];
    double rms = 0.;
    int i, j, i_rms = 0;

    srand(1);

    // Initialize weights and biases
    for (i = 0; i < HIDDENSIZE; i++) {
        bias_hidden[i] = ((double)rand() / RAND_MAX) - 0.5;
        for (j = 0; j < INPUTSIZE; j++) {
            weight_ih[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    for (i = 0; i < OUTPUTSIZE; i++) {
        bias_output[i] = ((double)rand() / RAND_MAX) - 0.5;
        for (j = 0; j < HIDDENSIZE; j++) {
            weight_ho[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    // Training
    for (i = 0; i < LOOPNUM; i++) {
        for (j = 0; j < data_size; j++) {
            input[0] = iris_data[j].sepal_length;
            input[1] = iris_data[j].sepal_width;
            input[2] = iris_data[j].petal_length;
            input[3] = iris_data[j].petal_width;
            input[4] = 1.0;  // Bias neuron 
            Forward(input, weight_ih, weight_ho, bias_hidden, bias_output, hidden, output);
            double target[OUTPUTSIZE] = {0.0, 0.0, 0.0};
            target[iris_data[j].species] = 1.0;
            Backward(input, hidden, output, target, weight_ih, weight_ho, bias_hidden, bias_output, 0.01);
        }
    }

    // Testing
    printf("Results for %s:\n", operation);
    int correct_predictions = 0;
    for (int j = 0; j < data_size; j++) {
        input[0] = iris_data[j].sepal_length;
        input[1] = iris_data[j].sepal_width;
        input[2] = iris_data[j].petal_length;
        input[3] = iris_data[j].petal_width;
        input[4] = 1.0;  // Bias neuron
        Forward(input, weight_ih, weight_ho, bias_hidden, bias_output, hidden, output);
        int predicted = 0;
        for (int i = 1; i < OUTPUTSIZE; i++) {
            if (output[i] > output[predicted]) {
                predicted = i;
            }
        }
        if (predicted == iris_data[j].species) {
            correct_predictions++;
        }
        rms += (iris_data[j].species - predicted) * (iris_data[j].species - predicted);
        i_rms++;
        //printf("INPUT: %.2f %.2f OUTPUT: %d EXPECTED: %d\n", iris_data[j].sepal_length, iris_data[j].sepal_width, predicted, iris_data[j].species);
    }
    //double result_rms = sqrt(rms/i_rms);
    //printf("RMS for %s: %f\n", operation, result_rms);
    double accuracy = (double)correct_predictions / data_size;
    printf("Accuracy for %s: %.2f%%\n\n", operation, accuracy * 100);

    return accuracy;
}

void k_fold_cross_validation(Iris iris_data[], int data_size) {
    int fold_size = data_size / K;
    double overall_rms = 0.0;

    for (int k = 0; k < K; k++) {
        printf("Fold %d\n", k+1);

        // Split the data into training and testing based on the current fold
        Iris train_data[data_size - fold_size];
        Iris test_data[fold_size];

        int train_idx = 0, test_idx = 0;
        for (int i = 0; i < data_size; i++) {
            if (i >= k * fold_size && i < (k+1) * fold_size) {
                test_data[test_idx++] = iris_data[i];
            } else {
                train_data[train_idx++] = iris_data[i];
            }
        }

        // Train and test using the split data
        double rms = train_and_test(train_data, data_size - fold_size, "Training");
        overall_rms += rms;
        rms = train_and_test(test_data, fold_size, "Testing");
        overall_rms += rms;
    }
}

int main() {
    Iris iris_data[150];
    int data_size = read(iris_data);

    if (data_size > 0) {
        k_fold_cross_validation(iris_data, data_size);
    }

    return 0;
}

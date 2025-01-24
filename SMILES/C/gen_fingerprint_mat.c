#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_FINGERPRINTS 33668
#define MAX_LENGTH 2050

// Function to calculate Jaccard distance
double jaccard_distance(const char *fp1, const char *fp2) {
    int intersection = 0;
    int union_count = 0;

    for (int i = 0; fp1[i] != '\0' && fp2[i] != '\0'; i++) {
        if (fp1[i] == '1' || fp2[i] == '1') {
            union_count++;
            if (fp1[i] == '1' && fp2[i] == '1') {
                intersection++;
            }
        }
    }

    return 1.0 - ((double)intersection / union_count);
}

double cosine_distance(char *smile1, char *smile2){
    int intersection = 0;
    int sum1 = 0;
    int sum2 = 0;
    
    for(int i=0; i<strlen(smile1); i++){
        if (smile1[i] == '1' || smile2[i] == '1' ){
            if (smile1[i] == '1' && smile2[i] == '1'){
                intersection++;
                sum1++;
                sum2++;
            }
            else{
                if (smile1[i] == '1'){
                    sum1++;
                }
                else{
                    sum2++;
                }
            }
        }
    }
    double theta = intersection / (double)(sum1 * sum2);
    
    return 1.0 - cos(theta);
}

// Main pour caculer la matrice de similarité
int main() {
    FILE *file = fopen("../data/morgan_fingerprints.txt", "r");
    if (!file) {
        perror("Failed to open inputfile");
        return EXIT_FAILURE;
    }

    char *fingerprints[MAX_FINGERPRINTS];
    char buffer[MAX_LENGTH];
    int count = 0;

    // Read fingerprints from file
    while (fgets(buffer, MAX_LENGTH, file) && count < MAX_FINGERPRINTS) {
        buffer[strcspn(buffer, "\n")] = '\0'; // Remove newline character
        fingerprints[count] = strdup(buffer);
        count++;
    }
    fclose(file);

     // Open the output file for writing
    FILE *output_file = fopen("../data/matrix/matrix_fingerprint_cos.txt", "w");
    if (!output_file) {
        perror("Failed to open output file");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < count; i++) {
        if (i % (int)(MAX_FINGERPRINTS / 100) == 0){
            printf("Progress: %d%%\n", i / (MAX_FINGERPRINTS / 100));
        }
        for (int j = i + 1; j < count - 1; j++) {
            double distance = cosine_distance(fingerprints[i], fingerprints[j]);
            fprintf(output_file, "%f,", distance);
        }
        double distance = cosine_distance(fingerprints[i], fingerprints[count - 1]);
        fprintf(output_file, "%f\n", distance);
    }

    // Free memory
    for (int i = 0; i < count; i++) {
        free(fingerprints[i]);
    }

    return EXIT_SUCCESS;
}
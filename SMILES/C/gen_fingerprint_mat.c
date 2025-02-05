#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cJSON.h"

#define MAX_FINGERPRINTS 30
#define MAX_LENGTH 2050
#define ITER_LIM 1000

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

    if (sum1 == 0 || sum2 == 0){
        return 1.0;
    }

    double theta = intersection / (double)(sqrt(sum1) * sqrt(sum2));
    
    return 1.0 - theta;
}

double cosine_distance_centroid(char *smile, double *centroid){
    double intersection = 0;
    int sum1 = 0;
    double sum2 = 0;
    
    for(int i=0; i<strlen(smile); i++){
        if (smile[i] == '1' || centroid[i] != 0 ){
            if (smile[i] == '1' && centroid[i] != 0.0){
                intersection += centroid[i];
                sum1++;
                sum2 += centroid[i]*centroid[i];
            }
            else{
                if (smile[i] == '1'){
                    sum1++;
                }
                else{
                    sum2 += centroid[i]*centroid[i];
                }
            }
        }
    }

    if (sum1 == 0 || sum2 == 0.0){
        return 1.0;
    }

    double theta = intersection / (double)(sqrt(sum1) * sqrt(sum2));
    
    return 1.0 - theta;
}

int CLS(const char *smile1, const char *smile2) {
    int m = strlen(smile1);
    int n = strlen(smile2);
    int dp[m + 1][n + 1];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            dp[i][j] = 0;
        }
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (smile1[i - 1] == smile2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = (dp[i - 1][j] > dp[i][j - 1]) ? dp[i - 1][j] : dp[i][j - 1];
            }
        }
    }

    return dp[m][n];
}

int **create_matrix(int rows, int cols) {
    int **matrix = (int **)malloc(rows * sizeof(int *));
    if (matrix == NULL) {
        perror("Failed to allocate memory for matrix");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int *)malloc(cols * sizeof(int));
        if (matrix[i] == NULL) {
            perror("Failed to allocate memory for matrix row");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

double **create_matrix_double(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        perror("Failed to allocate memory for matrix");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            perror("Failed to allocate memory for matrix row");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void free_matrix_int(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

cJSON* create_json(int **clusters, int K){
    cJSON *root = cJSON_CreateObject();
    char name[20];
    if (!root) {
        perror("Failed to create JSON object");
    }
    cJSON *cluster = NULL;
    for (int i=0; i<K; i++){
        cluster = cJSON_CreateArray();
        if (!cluster) {
            perror("Failed to create JSON array");
        }
        sprintf(name, "cluster%d", i);
        cJSON_AddItemToObject(root, name, cluster);
        for (int j=0; j<MAX_FINGERPRINTS; j++){
            if (clusters[i][j] == 1){
                cJSON_AddItemToArray(cluster, cJSON_CreateNumber(j));
            }
        }
    }
    return root;
}

cJSON* create_json_silhouette(double *sil, int size){
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        perror("Failed to create JSON object");
    }
    cJSON *silhouettes = NULL;
    silhouettes = cJSON_CreateArray();
    if (!silhouettes) {
        perror("Failed to create JSON array");
    }
    cJSON_AddItemToObject(root, "silhouettes", silhouettes);
    for (int i=0; i<size; i++){
        cJSON_AddItemToArray(silhouettes, cJSON_CreateNumber(sil[i]));
    }
    return root;
}

void init_centroid(double **centroid, char *fingerprints[], int K){
    int i_rand = 0;
    int current = 0;
    for (int i=0; i<K; i++){
        i_rand = rand() % MAX_FINGERPRINTS;
        for (int j=0; j<MAX_LENGTH-2; j++){
            current = fingerprints[i_rand][j] - '0';
            centroid[i][j] = (double)current;
        }
    }
}

void init_clusters(int **clusters, int K){
    int rand_K = 0;
    for (int i=0; i<MAX_FINGERPRINTS; i++){
        rand_K = rand() % K;
        clusters[rand_K][i] = 1;
    }
}

int find_cluster(int **clusters, int index, int K){
    for (int i=0; i<K; i++){
        if (clusters[i][index] == 1){
            return i;
        }
    }
    return -1;
}

void update_curr_centroid(double *curr_centroid, char *fp){
    for (int i=0; i<MAX_LENGTH-2; i++){
        if (fp[i] == '1'){
            curr_centroid[i]++;
        }
    }
}

void update_centroid(double **centroid, int **clusters, char *fingerprints[], int K){
    double *curr_centroid = (double *)malloc((MAX_LENGTH-2) * sizeof(double));;
    int count = 0;
    for (int i=0; i<K; i++){
        for (int j=0; j<MAX_FINGERPRINTS; j++){
            if (clusters[i][j] == 1){
                count++;
                update_curr_centroid(curr_centroid, fingerprints[i]);
            }
        }

        for (int k=0; k<MAX_FINGERPRINTS; k++){
            curr_centroid[k] /= count;
        }
        memcpy(centroid[i], curr_centroid, (MAX_LENGTH-2) * sizeof(int));   
    }
    free(curr_centroid);
}

int **k_mean_clustering(char *fingerprints[], int K){

    int **clusters = create_matrix(K, MAX_FINGERPRINTS);
    double **centroid = create_matrix_double(K, MAX_LENGTH-2);

    init_centroid(centroid, fingerprints, K);
    init_clusters(clusters, K);

    int change = 1;
    int count = 0;
    int num_cluster = 0;
    int count_change = 0;

    while (change && count < ITER_LIM){
        count++;
        change = 0;
        count_change = 0;

        for (int i=0; i<K; i++){
            for (int j=0; j<MAX_FINGERPRINTS; j++){
                num_cluster = find_cluster(clusters, j, K);
                if (num_cluster != -1){
                    if (num_cluster != i){
                        if (cosine_distance_centroid(fingerprints[j], centroid[i]) < cosine_distance_centroid(fingerprints[j], centroid[num_cluster])){
                            clusters[i][j] = 1;
                            clusters[num_cluster][j] = 0;
                            change = 1;
                            count_change++;
                        }
                    }
                }
                else {
                    perror("Failed to find cluster");
                }
            }
        }
        update_centroid(centroid, clusters, fingerprints, K);
        printf("Iteration #%d\n", count);
        printf("%d changements\n", count_change);
    }
    free_matrix(centroid, K);

    return clusters;
}

double mean_dist_within_cluster(char *fingerprint, int *cluster, char *fingerprints[]){
    double dist = 0;
    int count = 0;
    for (int i=0; i<MAX_FINGERPRINTS; i++){
        if (cluster[i] == 1){
            dist += cosine_distance(fingerprint, fingerprints[i]);
            count++;
        }
    }
    if (count == 1){
        return 0.0;
    }
    return dist / (count-1) ;    
}

double mean_dist_with_closest_cluster(char *fingerprint, int **clusters, char *fingerprints[], int num_cluster, int K){
    double dist = 0;
    double curr_dist = 0;
    int count = 0;
    for (int i=0; i<K; i++){
        if (i != num_cluster){
            for (int j=0; j<MAX_FINGERPRINTS; j++){
                if (clusters[i][j] == 1){
                    curr_dist += cosine_distance(fingerprint, fingerprints[j]);
                    count++;
                }
            }
            curr_dist /= count;
            if (curr_dist < dist || dist == 0){
                dist = curr_dist;
            }
        }
        curr_dist = 0;
        count = 0;
    }
    return dist ;  
}

double max(double a, double b) {
    return (a > b) ? a : b;
}

double compute_silhouette(int **clusters, char *fingerprints[], int K){
    int num_cluster = 0;
    double curr_a = 0;
    double curr_b = 0;
    double mean_silhouette = 0;
    for (int i = 0; i<MAX_FINGERPRINTS; i++){
        num_cluster = find_cluster(clusters, i, K);
        curr_a = mean_dist_within_cluster(fingerprints[i], clusters[num_cluster], fingerprints);
        curr_b = mean_dist_with_closest_cluster(fingerprints[i], clusters, fingerprints, num_cluster, K);
        
        mean_silhouette += (curr_b - curr_a) / max(curr_a, curr_b);

        /* if (i % (int)(MAX_FINGERPRINTS / 100) == 0){
            printf("Silhouette computing %d%%\n", i / (int)(MAX_FINGERPRINTS / 100));
            printf("Valeur de mean_silhouette : %f\n", mean_silhouette);
        } */
    }

    return mean_silhouette / MAX_FINGERPRINTS;
}

// Main pour caculer la matrice de similarité
int _main_simil() {
    FILE *file = fopen("../data/smiles_without_cn.txt", "r");
    if (!file) {
        perror("Failed to open input file");
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
            int distance = cosine_distance(fingerprints[i], fingerprints[j]);
            fprintf(output_file, "%d,", distance);
        }
        int distance = cosine_distance(fingerprints[i], fingerprints[count - 1]);
        fprintf(output_file, "%d\n", distance);
    }

    // Free memory
    for (int i = 0; i < count; i++) {
        free(fingerprints[i]);
    }

    return EXIT_SUCCESS;
}

char **load_fingerprints(char filepath[]){
    FILE *file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open input file");
    }

    char **fingerprints = (char **)malloc(MAX_FINGERPRINTS * sizeof(char *));
    for (int i = 0; i<MAX_FINGERPRINTS; i++){
        fingerprints[i] = (char *)malloc(MAX_LENGTH * sizeof(char));
    }
    char buffer[MAX_LENGTH];
    int count = 0;

    // Read fingerprints from file
    while (fgets(buffer, MAX_LENGTH, file) && count < MAX_FINGERPRINTS) {
        buffer[strcspn(buffer, "\n")] = '\0'; // Remove newline character
        fingerprints[count] = strdup(buffer);
        count++;
    }
    fclose(file);
    return fingerprints;
}

// Main pour faire un clustering k-mean
double _main_k_mean(int K, char **fingerprints){

    printf("Début du clustering\n");
    int **clusters = k_mean_clustering(fingerprints, K);

    double silhouette = compute_silhouette(clusters, fingerprints, K);
    printf("Silhouette pour K=%d : %f\n", K, silhouette);

    cJSON *results = create_json(clusters, K);
    printf("Tag 1\n");

    char *json_string = cJSON_Print(results);
    printf("Tag 2\n");

    // Open the output file for writing
    char name[40];
    sprintf(name, "../data/json/kmean_cosineFP_K%d.json", K);
    FILE *output_file = fopen(name, "w");
    if (!output_file) {
        perror("Failed to open output file");
        return EXIT_FAILURE;
    }

    printf("Tag3\n");
    cJSON_Delete(results);

    printf("Tag 4\n");
    free_matrix_int(clusters, K);

    printf("Tag 5\n");
    fprintf(output_file, "%s\n", json_string);
    free(json_string);

    printf("Tag 6\n");
    fclose(output_file);

    printf("Tag 7\n");
    return silhouette;
}

void find_best_k(int MAX_K, char **fingerprints){
    double *sil = (double *)malloc(MAX_K * sizeof(double));

    for (int i = 2; i<MAX_K+1; i++){
        sil[i] = _main_k_mean(i, fingerprints);
    }

    cJSON *results_sil = create_json_silhouette(sil, MAX_K);
    char *json_string = cJSON_Print(results_sil);
    FILE *output_file = fopen("../data/json/silhouettes.json", "w");
    if (!output_file) {
        perror("Failed to open output file");
    }

    fprintf(output_file, "%s\n", json_string);
    fclose(output_file);
    free(sil);
}

void free_fingerprints(char **fingerprints, int count){
    for (int i = 0; i < count; i++) {
        free(fingerprints[i]);
    }
    free(fingerprints);
}

int main(){
    //int i = _main_simil();
    //int i = _main_k_mean();
    char fp_1[] = "../data/[2M+Ca]2+_fp.txt";
    char fp_2[] = "../data/[M-3H2O+H]1+_fp.txt";
    char fp_3[] = "../data/[M+Ca]2+_fp.txt";
    char **fingerprints = load_fingerprints(fp_1);
    find_best_k(17, fingerprints);
    free_fingerprints(fingerprints, MAX_FINGERPRINTS);
    return 0;
}
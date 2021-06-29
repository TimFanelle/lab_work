#include <iostream>
#include <cstdio>

int main(){
    FILE *fpt;
    fpt = fopen("testandMest00.csv", "w+");

    float i[3]; //define array that will hold the outputs
    //a bit more concise
    /*
    for(int p = 0; p<20; p++){
        for(int j=0; j < 3; j++){
            i[j] = (float)rand()/(float)RAND_MAX * (float)(3); //fill the array with random numbers between [0-3]
        }
        fprintf(fpt, "%d, %f, %f, %f\n", p, i[0], i[1], i[2]); //write array values to the file
    }
    */

    
    //you can also create the array and then do this
    int numLines = 25;
    int arrLen = sizeof(i)/sizeof(i[0]);
    for(int p = 0; p<numLines; p++){
        for(int j=0; j < arrLen; j++){
            i[j] = (float)rand()/(float)RAND_MAX * (float)(3); //fill the array with random numbers between [0-3]
        }
        fprintf(fpt, "%d, ", p);
        for(int k = 0; k<arrLen; k++){
            if(k==arrLen-1){
                fprintf(fpt, "%f\n", i[k]);
            }
            else{
                fprintf(fpt, "%f, ", i[k]);
            }
        }
    }
    

    fclose(fpt); //close file
    
    return 0;
}

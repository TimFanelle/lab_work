#include <stdio.h>
//#include <string.h>

#define BUFFER_SIZE 1024 //setting max number of characters to be read from a line

//use this if <string.h> is not included, but not if string.h is
void *memcopy(void *dest, const void *src, size_t n){
    size_t i = 0;
    char* newsrc = (char*)src;
    char* newdest = (char*)dest;
    while(i<n){
        newdest[i]=newsrc[i];
        i++;
    }
    return newdest;
}

int main(){
    
    FILE *fpt;
    fpt = fopen("testandMest.csv", "r"); //opening the CSV file

    float temp[100][4] = {{ 0.0 }}; //define a temparary array that is longer than the expected length of the file
    char buf[BUFFER_SIZE]; //define a buffer array to hold the line when read from the file
    int label;
    float i[4];

    int count = 0; //count how many lines are read from the file
    int c;
    for ( ; ; ){
        c = getc(fpt); //grab next character
        if (c == EOF){ //check if the next character is the end of file
            break;
        }
        fseek(fpt, -1L, SEEK_CUR); //go back one character
        fgets(buf, sizeof(buf), fpt); //read next line in file
        sscanf(buf, "%d, %f, %f, %f", &label, &i[1], &i[2], &i[3]); //break line into its various parts
        i[0] = (float)label; //convert integer label to float so it can be saved as such

        //memcpy(temp[count], i, sizeof(i)); //copy data to temp array
        memcopy(temp[count], i, sizeof(i));
        
        ++count;
    }
    fclose(fpt); //close file

    float final_size_inpts[count][4]; //define array that is the exact length needed
    for(int j = 0; j<count; j++){
        //memcpy(final_size_inpts[j], temp[j], sizeof(temp[j])); //copy data from temp to final
        memcopy(final_size_inpts[j], temp[j], sizeof(temp[j]));
    }

    //define rows and columns variables so they don't need to be calculated more than once
    int rows = sizeof(final_size_inpts)/sizeof(final_size_inpts[0]);
    int columns = sizeof(final_size_inpts[0])/sizeof(final_size_inpts[0][0]);
    
    //check if expected actions happened
    printf("%d::%d\n", rows, columns);
    for(int j = 0; j<rows; j++){
        for(int k = 0; k<columns; k++){
            if(k == 0){
                printf("%d ", (int)final_size_inpts[j][k]);
            }
            else{
                printf("%f ", final_size_inpts[j][k]);
            }
        }
        printf("\n");
    }  

    return 0;
}

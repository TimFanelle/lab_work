// quickDAQ.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <Windows.h>
#include <quickDAQ.h>
#include <stdlib.h>
#include <time.h>

#include <string.h> // included solely for the memcpy function
#include <cstdio>

#define BUFFER_SIZE 1024 //setting max number of characters to be read from a line
#define muscles 12

//using namespace std;
int main()
{
	/*
		unsigned input;
		printf_s("Hello World! Enter a number (0-9): ");
		input = getchar() - '0';
		printf_s("\nThe number you input was %d", input);
		printf_s("\nPress a key to continue...\n");
		getchar();
	*/

	//opening the csv files
	FILE* activations;
	activations = fopen("motor_activations.csv", "r+");
	//fopen_s(&activations, "motor_activations.csv", "r+"); //input file


	//fopen_s(&activations_and_angles, "timeAndAngles.csv", "w"); //output file

	//setting up reading the inputs
	float temp[100][muscles] = { { 0.0 } }; //define a temporary array that is longer than the expected length of the file
	char buf[BUFFER_SIZE]; //define a buffer array to hold the line when read from the csv file
	float i[muscles];

	//read the inputs

	int count = 0; //set to read how many lines were in the file
	int c;
	for (;;) {
		c = getc(activations); //grab next character		
		if (c == EOF) { //check if that is the end of the file
			break;
		}
		fseek(activations, -1L, SEEK_CUR); //go back one character
		fgets(buf, sizeof(buf), activations); //read next line in the file
		sscanf_s(buf, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f", &i[0], &i[1], &i[2], &i[3], &i[4], &i[5], &i[6], &i[7], &i[8], &i[9], &i[10], &i[11]); // break the line into its various parts
		memcpy(temp[count], i, sizeof(i)); //copy data to temp array

		++count;
	}/*
	float64 final_size_inputs[count][muscles]; //define array that is the correct length needed
	for (int j = 0; j < count; j++) {
		memcpy(final_size_inputs[j], temp[j], sizeof(temp[j]));
	}
	*/
	fclose(activations); //close file

	FILE* activations_and_angles;
	activations_and_angles = fopen("timeAndAngles.csv", "w");

	// initialize
	int iterations = count * 1000; //define how many iterations will occur

	srand((unsigned)time(0));
	quickDAQinit();

	// configure channels and sample clock
	pinMode(5, ANALOG_IN, 0);
	for (int i = 0; i < 12; i++) {
		pinMode(2, ANALOG_OUT, i);
	}
	pinMode(2, ANALOG_OUT, 16);
	pinMode(2, ANALOG_OUT, 17);
	pinMode(2, ANALOG_OUT, 18);
	pinMode(2, ANALOG_OUT, 19);

	pinMode(2, DIGITAL_OUT, 0);

	for (int i = 0; i < 8; i++) {
		pinMode(3, CTR_ANGLE_IN, i);
	}

	setSampleClockTiming((samplingModes)HW_CLOCKED/*DAQmxSampleMode*/, DAQmxSamplingRate, DAQmxClockSource, (triggerModes)DAQmxTriggerEdge, DAQmxNumDataPointsPerSample, TRUE);

	printf("\nIO timeout is %f\n", DAQmxDefaults.IOtimeout);

	// read/write data
	float64 AI;
	float64 mtrCmd[16] = { 0,0,0, 0,0,0, 0,0,0, 0,0,0,	0,0,0,0 };
	const float64 muscleTone = 0.2;
	uInt32			DO1 = 0x000000ff;
	const float64	DO2[4] = { 5,5,5,5 };
	float64 CI[8] = { 0,0, 0,0, 0,0, 0,0 };

	//printf("\nData to be written: AO: %lf, DO: %lX\n", (double)AO, DO);
	printf("Press enter to start control of hand");
	getchar();

	// start tasks
	quickDAQstart();


	syncSampling(); //wait for HW timed sample

	/*
	readAnalog(5, &AI);
	printf("pass AI\n");

	writeAnalog(2, &AO);
	printf("pass AO\n");

	writeDigital(2, &DO);
	printf("pass DO\n");

	readCounterAngle(3, 0, &CI);
	printf("\nData read: CI: %lf\n", (double)CI);
	*/

	// Enable motor amps
	readAnalog(5, &AI);
	for (int i = 12; i < 16; i++) {
		mtrCmd[i] = DO2[i - 12];
	}
	writeDigital(2, &DO1); // enable first 8 motor amplifiers
	writeAnalog(2, &(mtrCmd[0])); // enable 9-12th motor amps
	printf("Motor Enabled\n");

	// provide muscle tone - ensures no tendon/cable is slack
	for (int i = 0; i < 12; i++) {
		mtrCmd[i] = muscleTone;
	}
	writeAnalog(2, &(mtrCmd[0]));
	printf("Motor Wound up\n");

	// Control Loop
	for (unsigned long t = 0; t < iterations; t++) {
		syncSampling();
		readAnalog(5, &AI);

		// change motor current/activations once every 1000 iterations
		if (t % 1000 == 0) {
			for (int i = 0; i < 12; i++) {
				mtrCmd[i] = temp[t / 1000][i] + muscleTone; //final_size_inputs[i] //(float((rand() % 8)) / 10) + muscleTone;
			}
			writeAnalog(2, &(mtrCmd[0]));
			printf("\n\nMOTOR: %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f\n\n",
				mtrCmd[0], mtrCmd[1], mtrCmd[2], mtrCmd[3], mtrCmd[4], mtrCmd[5], mtrCmd[6], mtrCmd[7], mtrCmd[8], mtrCmd[9], mtrCmd[10], mtrCmd[11]);
		}

		// read counter angle every iteration
		for (int j = 0; j < 8; j++) {
			readCounterAngle(3, j, &(CI[j]));
		}
		printf("ANGLE: %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\r",
			CI[0], CI[1], CI[2], CI[3], CI[4], CI[5], CI[6], CI[7]);

		//write all values to the file
		fprintf(activations_and_angles, "%3.3f, %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f, %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n", float(float(t) / 1000) + .001, mtrCmd[0], mtrCmd[1], mtrCmd[2], mtrCmd[3], mtrCmd[4], mtrCmd[5], mtrCmd[6], mtrCmd[7], mtrCmd[8], mtrCmd[9], mtrCmd[10], mtrCmd[11], CI[0], CI[1], CI[2], CI[3], CI[4], CI[5], CI[6], CI[7]);

	}
	fclose(activations_and_angles); //close file

	syncSampling();
	readAnalog(5, &AI);
	for (int i = 0; i < 16; i++) {
		mtrCmd[i] = 0;
	}
	writeAnalog(2, &(mtrCmd[0]));
	printf("\n\nMotor wound down\n");

	DO1 = 0x00000000;
	writeDigital(2, &(DO1));
	printf("Motor disabled\n\n");

	syncSampling();

	// end tasks
	quickDAQstop();

	// Terminate library
	quickDAQTerminate();

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
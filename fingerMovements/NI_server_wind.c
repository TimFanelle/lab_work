// quickDAQ.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <stdio.h>
#include <Windows.h>
#include <quickDAQ.h>
#include <stdlib.h>
#include <time.h>

#include <string.h>

#include <winsock2.h>

#define muscles 3
#define encodes 3
#define PORT 4268
#define ADDRESS "127.0.0.1"
#define MAX_CLIENT_MSG_LEN 1024

char buffer[1024];
char recvBuffer[1024];
float activations[muscles];
float encoders[encodes];
float potential = 0.0;
char obs[1024];
char* token;

void readEncoders(){
    char smallBuf[20];
    strcpy(obs, "");
    for(int i = 0; i<encodes; i++){
        gcvt(encoders[i], 16, smallBuf);
        if(i>0){
            strcat(obs, ", ");
        }
        strcat(obs, smallBuf);
    }
    gcvt(potential, 16, smallBuf);
    strcat(obs, ", ");
    strcat(obs, smallBuf);
}

//using namespace std;
int main()
{

	// initialize
	//int iterations = count * 1000; //define how many iterations will occur

	srand((unsigned)time(0));
	quickDAQinit();

	// configure channels and sample clock
	pinMode(5, ANALOG_IN, 0);
	for (int i = 0; i < muscles; i++) {
		pinMode(2, ANALOG_OUT, i);
	}
	pinMode(2, ANALOG_OUT, 16);
	pinMode(2, ANALOG_OUT, 17);
	pinMode(2, ANALOG_OUT, 18);
	pinMode(2, ANALOG_OUT, 19);

	pinMode(2, DIGITAL_OUT, 0);

	for (int i = 0; i < encodes; i++) {
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

    // START SERVER

    /* Initializing server socket */
    //int welcomeSocket, newSocket;
    int option = 0;
    /*
    struct sockaddr_in serverAddr;
    struct sockaddr_storage serverStorage;
    socklen_t addr_size;

    welcomeSocket = socket(PF_INET, SOCK_STREAM, 0);

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(4268);
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

    memset(serverAddr.sin_zero, '\0', sizeof serverAddr.sin_zero);

    bind(welcomeSocket, (struct sockaddr *) &serverAddr, sizeof serverAddr);
    */
    WSADATA wsa;
    SOCKET welcomeSocket, acceptSocket;
    struct sockaddr_in server, client;
    int sockAddrInLength = sizeof(struct sockaddr_in);
    char clientMessage[MAX_CLIENT_MSG_LEN];
    int clientMessageLength;
    char* retMessage;
    /* begin listening for client messages */
    WSAStartup(MAKEWORD(2,2), &wsa);

    welcomeSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    server.sin_addr.s_addr = inet_addr(ADDRESS);
    server.sin_family = AF_INET;
    server.sin_port = PORT;
    bind(welcomeSocket, (struct sockaddr*)&server, sizeof server);

    int q = 1;
    
    int n;
    if(listen(welcomeSocket, 5)==0){
        printf("Listening\n");
    }
    else{
        printf("Error\n");
    }

    for(;;){
        /* break infinite loop */
        end_loop:
            if(q == 0){
                break;
            }
        /* Receiving client data */
        //addr_size = sizeof serverStorage;
        newSocket = accept(welcomeSocket, (struct sockaddr *) &client, &sockAddrInLength);
        n = recv(newSocket, clientMessage, sizeof clientMessage, 0);
        
        /* sync */
        syncSampling(); //sync every time a connection is accepted
        readAnalog(5, &AI);

        /* break it into parts */
        int l = 0;
        token = strtok(clientMessage, ";");
        sscanf(token, "%d", &option);

        /* determine action */
        switch(option){
            case 1:
                /* assign muscle activations and save encoder values and slider value */
                token = strtok(NULL, ";");
                while(token != NULL){
                    sscanf(token, "%f", &activations[l]);
                    token = strtok(NULL, ";");
                    l++;
                }
                printf("\n\nMOTOR:");
                for(int i = 0; i < muscles; i++){
                    mtrCmd[i] = activations[i] + muscleTone;
                    printf(" %3.1f", mtrCmd[i]);
                }
                printf("\n\n");
                writeAnalog(2, &(mtrCmd[0]));
                for(int j = 0; j<encodes; j++){
                    readCounterAngle(3, j, &(CI[j]));
                    encoders[j] = CI[j];
                }
                //TODO: read potentiameter
                readEncoders();
                break;
            case 2:
                /* take the outputs from the encoders and slider and save them*/
                for(int j = 0; j<encodes; j++){
                    readCounterAngle(3, j, &(CI[j]));
                    encoders[j] = CI[j];
                }
                //TODO: read potentiameter
                readEncoders();
                break;
            case 3:
                /* break loop, shut down server, and finish running program*/
                send(acceptSocket, "Finished", 9, 0);
                printf("Completed and closed");
                q = 0;
                goto end_loop;
            default:
                break;
        }
        /* Send encoder and slider values back to client */
        printf("%s\n", obs);
        //strcpy(buffer, obs);
        len = strlen(obs);
        send(acceptSocket, obs, len, 0);

        sleep(1);
    }
    closesocket(welcomeSocket);
    WSACleanup();
    // END SERVER

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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <string.h>

//multi-process experiments.
//we use the fork-exec methods on darknet inferencing.
//1.create shared memory space and initialize darknet processes.
//2.each processes wait until initilization of network is all finished.(synched environment)
//3.start every inference concurrently.
//4.collect informantion of inference time on shared memory and exit the program.

//external variables
int *shm_ptr; //shared memory pointer.

int main (int argc, char* argv[]){

    int n = 1;                  //number of child process.
    pid_t pid[n];               //process id array.
    int status;                 //checking child exit status.
    int i;                      //iteration.
    key_t SHARED_MEMORY_KEY[n]; //shm keys for child process.
    int shmid[n];                 //shared memory id.
    printf("manager program starting\n");
    
    //1-0.initialize shared mem for each process.
    for (i=0; i<n; i++){
        SHARED_MEMORY_KEY[n] = i+1000;
        shmid[i] = shmget(SHARED_MEMORY_KEY[i],sizeof(int)*2,0666|IPC_CREAT);
        if(shmid[i]==-1){
            perror("shmget failed\n");
            exit(0);
        }
    }
   

    //1-1.make n child processes.
    for (i=0; i<n; i++){
        
        //allocate the pointer to each shm.(+testcode)
        shm_ptr = (int*)shmat(shmid[i],NULL,0);
        shm_ptr[0] = 0; //Inference sign.
        shm_ptr[1] = 0; //model ready sign.
        //save shmid as character for argv paramater.
        char temp[32];
        sprintf(temp,"%d",shmid[i]);

        //fork and exec.
        pid[i] = fork();

        if(pid[i] == 0){
            //open file to write output & read input
            int fd;
            int fd2;
            char output_idx[4];
            char output_name[50];
            sprintf(output_idx,"%d",i);
            strcpy(output_name,"res_multi_");
            strcat(output_name,output_idx);
            if((fd = open(output_name, O_RDWR | O_CREAT))==-1){
                perror("open");
                return 1;
            }
            dup2(fd,STDOUT_FILENO);
            dup2(fd,STDERR_FILENO);
            close(fd);

            if((fd2 = open("input.txt", O_RDONLY))==-1){
                perror("open");
                return 1;
            }
            dup2(fd2,STDIN_FILENO);
            close(fd2);
            //!opening files.
            if ((i%2)==0){
                execl("/home/nvidia/darknet_resource/darknet", "darknet", "detector","test",
                    "/home/nvidia/darknet_resource/cfg/coco.data",
                    "/home/nvidia/darknet_resource/yolov3-tiny.cfg",
                    "/home/nvidia/darknet_resource/yolov3-tiny.weights",
                    "-dont_show","-ext_output",
                    argv[1],temp, NULL);
            }
            else{
                execl("/home/nvidia/darknet_resource/darknet", "darknet", "detector","test",
                    "/home/nvidia/darknet_resource/cfg/coco.data",
                    "/home/nvidia/darknet_resource/yolov3-tiny.cfg",
                    "/home/nvidia/darknet_resource/yolov3-tiny.weights",
                    "-dont_show","-ext_output",
                    argv[2],temp, NULL);
            }


            printf("oh, process is not executed...\n");
        }
    }
    
    //4. collect child process information.
    //testcode. spinlock until child gives certain infor
    
    /*
    for (i=0; i<1; i++){
        shm_ptr = (int*)shmat(shmid[i],NULL,0);
        while(shm_ptr[1] == 0){ 
            //wait until all model is ready
        }
        printf("successfully exited!\n");
    }
    
    for (i=0; i<1; i++){
        //allow inference
        shm_ptr = (int*)shmat(shmid[i],NULL,0);
        shm_ptr[0] = 1; 
        printf("manager : adjusted value to %d\n",shm_ptr[0]);
    }
    
    for (i=0; i<1; i++){
        shm_ptr = (int*)shmat(shmid[i],NULL,0);
        while(shm_ptr[1] == 1){}
    }
    */
    printf("end of manager.\n");
    return 0;
}


        

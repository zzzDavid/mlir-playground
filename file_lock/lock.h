#include <sys/file.h>
#include <fcntl.h>
#include "unistd.h"
#include <time.h>

char* currTime() {
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  return asctime (timeinfo);
}

int file_lock(char* lock_file)
{
    const char *lname = "Mutual Exlusive Lock (LOCK_EX)";
    int lock = LOCK_EX;
    int fd;

    if (access(lock_file, F_OK) == 0) {
        // file exists
        fd = open(lock_file, O_RDONLY);               /* Open file to be locked */
        if (fd == -1)
            printf("Cannot open lock file %s\n", lock_file); 
    } else {
        // file doesn't exist
        printf("Lock file %s doesn't exist\n", lock_file);
        return -1;
    }

    printf("PID %ld: requesting %s at %s", (long) getpid(), lname, currTime());

    if (flock(fd, lock) == -1) {
        if (errno == EWOULDBLOCK) {
            printf("PID %ld: already locked - bye!", (long) getpid());
            return -1;
        } else {
            printf("flock failed (PID=%ld)", (long) getpid());
            return -1;
        }
    }

    printf("PID %ld: granted    %s at %s", (long) getpid(), lname, currTime());

    return 0;
}


int file_unlock(char* lock_file)
{
    const char *lname = "Mutual Exlusive Lock (LOCK_EX)";
    int lock = LOCK_UN;
    int fd;

    if (access(lock_file, F_OK) == 0) {
        // file exists
        fd = open(lock_file, O_RDONLY);               /* Open file locked */
        if (fd == -1)
            printf("Cannot open lock file %s\n", lock_file); 
    } else {
        // file doesn't exist
        printf("Lock file %s doesn't exist\n", lock_file);
        return -1;
    }

    printf("PID %ld: releasing  %s at %s", (long) getpid(), lname, currTime());

    if (flock(fd, lock) == -1) {
        printf("flock failed (PID=%ld)", (long) getpid());
        return -1;
    }

    printf("PID %ld: released   %s at %s", (long) getpid(), lname, currTime());

    return 0;
}
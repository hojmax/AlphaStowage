#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Set the seed for random number generation to 0
    srand(0);

    // Generate and print a random number
    int randomNumber = rand();
    printf("Random number: %d\n", randomNumber);

    return 0;
}
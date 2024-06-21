#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>

double calculate(double operand1, char operators, double operand2) {   // Get the answer by directly using the operator.
    switch (operators) {
        case '+':
            return operand1 + operand2;
        case '-':
            return operand1 - operand2;
        case '*':
            return operand1 * operand2;
        case '/':
            return operand1 / operand2;
        default:
            printf("Error: Invalid operator '%c'.\n", operators);
            return -1;
    }
}

bool checkQuit(const char input[]) {  //ASCII value of `quit`.
    return (input[0] == 113) && (input[1] == 117) && (input[2] == 105) && (input[3] == 116);
}

bool checkValidOperator(char operators) {
    if ((operators == '+') || (operators == '-') || (operators == '*') || (operators == '/'))
        return true;
    else
        return false;
}

bool checkDividedZero(char operators, double operand2) {
    return (operand2 == 0) && (operators == '/');
}

bool checkBig(double operand1, double operand2) {  //Double type gradully loss of precision when gets bigger.
    return operand1 * operand2 >= 1e7;
}

void bigMul(const int a[], const int b[], int mulResult[], const int sizeA, const int sizeB) {

    for (int i = 0; i < sizeA + sizeB; i++)
        mulResult[i] = 0;

    for (int i = 0; i < sizeA; i++) {
        for (int j = 0; j < sizeB; j++)
            mulResult[i + j + 1] += a[i] * b[j];
    }

    for (int k = sizeA + sizeB - 1; k > 0; k--) {
        if (mulResult[k] >= 10) {
            mulResult[k - 1] += mulResult[k] / 10;   // Add with the quotation.
            mulResult[k] %= 10;    // Store with the reminder
        }
    }

}

int getLength(double operand) {  // Get the length of the array.
    int count = 1;

    if (operand == 0)
        return 1;

    while (operand >= 10) {
        operand /= 10;
        count++;
    }
    return count;
}

int main(int argc, char *argv[]) {
    double operand1, operand2, result;
    char operators;

    if (argc == 4) {
        operand1 = atof(argv[1]);
        operators = argv[2][0];
        operand2 = atof(argv[3]);
        if (checkDividedZero(operators, operand2))
            printf("Error: Division by zero\n");
        else if (checkBig(operand1, operand2) && operators == '*') { // The inputs is large.
            int sizeA = getLength(operand1);
            int sizeB = getLength(operand2);
            int a[sizeA], b[sizeB];
            for (int i = 0; i < sizeA; i++)   // Load array a and b from the command line.
                a[i] = argv[1][i] - 48;       // The difference between ASCII value and digit is 48.
            for (int i = 0; i < sizeB; i++)
                b[i] = argv[3][i] - 48;
            int mulRe[sizeA + sizeB];
            bigMul(a, b, mulRe, sizeA, sizeB);
            for (int i = 0; i < sizeA; i++)
                printf("%d",a[i]);
            printf(" * ");
            for (int i = 0; i < sizeB; i++)
                printf("%d",b[i]);  
            printf(" = ");
            for (int i = 0; i < sizeA + sizeB; i++) {
                if (i == 0 && mulRe[i] == 0) 
                    continue;
                printf("%d", mulRe[i]);
            }
            printf("\n");
            
        } else {  // Get the answer by directly using the operator.
            result = calculate(operand1, operators, operand2);
            printf("%.2lf %c %.2lf = %.4lf\n", operand1, operators, operand2, result);
        }

    } else if (argc == 1) {
        char input[1000];
        printf("Enter an expression (e.g., 2 + 3) or type 'quit' to exit:\n");
        while (true) {

            fgets(input, 1000, stdin);

            if (checkQuit(input))  // If type quit, then terminate.
                break;

            if (sscanf(input, "%lf %c %lf", &operand1, &operators, &operand2) == 3) {

                if (!checkValidOperator(operators))
                    printf("Error: Invalid operator '%c'.\n", operators);
                else if (((operand2 == 0) && (operators == '/')))
                    printf("Error: Division by zero\n");
                else if (checkBig(operand1, operand2) && operators == '*') {
                    int sizeA = getLength(operand1);
                    int sizeB = getLength(operand2);
                    int a[sizeA], b[sizeB];
                    for (int i = 0; i < sizeA; i++)   // Load array a and b from the original input.
                        a[i] = input[i] - 48;         // The difference between ASCII value and digit is 48.
                    for (int i = sizeA + 3; i < sizeB + sizeA + 3; i++)
                        b[i - sizeA - 3] = input[i] - 48;  // We have two spaces in the input.
                    int mulRe[sizeA + sizeB];
                    bigMul(a, b, mulRe, sizeA, sizeB);
                    for (int i = 0; i < sizeA; i++)
                        printf("%d",a[i]);
                    printf(" * ");
                    for (int i = 0; i < sizeB; i++)
                        printf("%d",b[i]);  
                    printf(" = ");
                    for (int i = 0; i < sizeA + sizeB; i++) {
                        if (i == 0 && mulRe[i] == 0) 
                            continue;
                        printf("%d", mulRe[i]);
                    }
                    printf("\n");   
                    
                } else {
                    result = calculate(operand1, operators, operand2);
                    printf("%.2lf %c %.2lf = %.4lf\n", operand1, operators, operand2, result);
                }

            } else
                printf("Error: Invalid number format.\n");

            printf("Enter another expression or type 'quit' to exit:\n");

        }

    } else {
        printf("Please enter the arguments as these two formats: %s <number1> <operator> <number2>\n", argv[0]);
        printf("or: %s (without any other arguments for interactive mode)\n", argv[0]);
        printf("If you want to express multiple sign in single mode, please type '*' instead of directly typing *.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

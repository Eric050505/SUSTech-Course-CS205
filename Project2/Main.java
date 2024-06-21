public class Main {
    public static void main(String[] args) {
        for (int i = 0; i <= 2000; i += 100) {
            float[][] A = createMatrix(i);
            float[][] B = createMatrix(i);
            float[][] C = new float[i][i];
            System.out.println("The Size of Matrices: " + i);
            long startTime = System.currentTimeMillis();
            multiplyMatrices(A, B, C, i);
            long endTime = System.currentTimeMillis();
            System.out.println("Time Taken: " + (endTime - startTime) + " ms");
            startTime = System.nanoTime();
            optimized_I(A, B, C, i);
            endTime = System.nanoTime();
            System.out.println("Time Taken after the first Optimization: " + (endTime - startTime) / 1e6 + " ms");
            startTime = System.nanoTime();
            optimized_II(A, B, C, i);
            endTime = System.nanoTime();
            System.out.println("Time Taken after the second Optimization: " + (endTime - startTime) / 1e6 + " ms");
            startTime = System.nanoTime();
            optimized_III(A, B, C, i);
            endTime = System.nanoTime();
            System.out.println("Time Taken after the third Optimization: " + (endTime - startTime) / 1e6 + " ms");
            System.out.println("------------------------------------------------\n");
        }
    }

    private static float[][] createMatrix(int N) {
        float[][] matrix = new float[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                matrix[i][j] = (float) Math.random();
        return matrix;
    }

    private static void multiplyMatrices(float[][] A, float[][] B, float[][] C, int N) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C[i][j] += A[i][k] * B[k][j];
    }

    private static void optimized_I(float[][] A, float[][] B, float[][] C, int N) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float r = 0;
                for (int k = 0; k < N; k++)
                    r += A[i][k] * B[k][j];
                C[i][j] = r;
            }
    }

    private static void optimized_II(float[][] A, float[][] B, float[][] C, int N) {
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                float r = A[i][k];
                for (int j = 0; j < N; j++)
                    C[i][j] += r * B[k][j];
            }
    }

    private static void optimized_III(float[][] A, float[][] B, float[][] C, int N) {
        int BLOCK_SIZE = 50;
        if (N > 1500) BLOCK_SIZE = 100;
        for (int ii = 0; ii < N; ii += BLOCK_SIZE)
            for (int kk = 0; kk < N; kk += BLOCK_SIZE)
                for (int i = ii; i < Math.min(ii + BLOCK_SIZE, N); i++)
                    for (int k = kk; k < Math.min(kk + BLOCK_SIZE, N); k++) {
                        float r = A[i][k];
                        for (int j = 0; j < N; j++)
                            C[i][j] += r * B[k][j];
                    }
    }

}

package numerical.anaylsis;

import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;
import java.util.Scanner;

public class Main {
    public SimpleMatrix createTriDiag(int n) {
        SimpleMatrix m = SimpleMatrix.identity(n);
        for (int i = 0; i < n; i++) {
            m.set(i, i, 4);
            if (i != n - 1) {
                m.set(i, i + 1, -1);
                m.set(i + 1, i, -1);
            }
        }
        return m;
    }

    public SimpleMatrix upper(SimpleMatrix m) {
        m = m.copy();
        int n = m.numCols();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                m.set(i, j, 0);
            }
        }
        return m;
    }

    public SimpleMatrix lower(SimpleMatrix m) {
        m = m.copy();
        int n = m.numCols();
        for (int j = 0; j < n; j++) {
            for (int i = 0; j >= i; i++) {
                m.set(i, j, 0);
            }
        }
        return m;
    }

    public SimpleMatrix diag(SimpleMatrix m) {
        m = m.copy();
        int n = m.numCols();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    m.set(i, j, 0);
                }
            }
        }
        return m;
    }

    public SimpleMatrix jacobi(SimpleMatrix A, SimpleMatrix b) {
        int n = A.numCols();
        SimpleMatrix X = SimpleMatrix.random_DDRM(n, 1, 0, 1, new Random());
        SimpleMatrix xc;
        SimpleMatrix U = upper(A);
        SimpleMatrix L = lower(A);
        SimpleMatrix D = diag(A);

        for (int k = 0; ; k++) {
            SimpleMatrix dInv = D.invert();
            SimpleMatrix m1 = dInv.scale(-1).mult(L.plus(U));
            m1 = m1.mult(X);
            SimpleMatrix m2 = dInv.mult(b);
            xc = m1.plus(m2);

            double xDelta = xc.minus(X).elementPower(2).elementSum();
            xDelta = Math.sqrt(xDelta);
            X = xc;
            if (xDelta <= 1e-4)
                break;
        }

        return X;
    }

    public SimpleMatrix gaussSeidel(SimpleMatrix A, SimpleMatrix b) {
        int n = A.numCols();
        SimpleMatrix X = SimpleMatrix.random_DDRM(n, 1, 0, 1, new Random());
        SimpleMatrix xc;
        SimpleMatrix U = upper(A);
        SimpleMatrix L = lower(A);
        SimpleMatrix D = diag(A);

        for (int k = 0; ; k++) {
            SimpleMatrix ldInv = L.plus(D).invert();
            SimpleMatrix m1 = ldInv.scale(-1).mult(U).mult(X);
            SimpleMatrix m2 = ldInv.mult(b);
            xc = m1.plus(m2);

            double xDelta = xc.minus(X).elementPower(2).elementSum();
            xDelta = Math.sqrt(xDelta);
            X = xc;
            if (xDelta <= 1e-4)
                break;
        }

        return X;
    }

    public SimpleMatrix sor(SimpleMatrix A, SimpleMatrix b, double w) {
        int n = A.numCols();
        SimpleMatrix X = SimpleMatrix.random_DDRM(n, 1, 0, 1, new Random());
        SimpleMatrix xc;
        SimpleMatrix U = upper(A);
        SimpleMatrix L = lower(A);
        SimpleMatrix D = diag(A);

        for (int k = 0; ; k++) {
            SimpleMatrix dlInv = L.scale(w).plus(D).invert();
            SimpleMatrix xM = D.scale(1 - w).minus(U.scale(w)).mult(X);
            SimpleMatrix m1 = dlInv.mult(xM);
            SimpleMatrix m2 = dlInv.scale(w).mult(b);
            xc = m1.plus(m2);

            double xDelta = xc.minus(X).elementPower(2).elementSum();
            xDelta = Math.sqrt(xDelta);
            X = xc;
            if (xDelta <= 1e-4)
                break;
        }

        return X;
    }

    public static void iterativeMethods() {
        Main myMain = new Main();
        int[] dims = new int[]{4, 16, 64, 256, 1024};
        for (int n : dims) {
            SimpleMatrix A = myMain.createTriDiag(n);
            SimpleMatrix b = SimpleMatrix.random_DDRM(n, 1, 1, 1, new Random());
            double w = 1.2;
            long time = System.currentTimeMillis();
            SimpleMatrix j = myMain.jacobi(A, b);
            long diff = System.currentTimeMillis() - time;
            System.out.printf("Jacobi Method for %dx%d matrix time: %f seconds\n", n, n, diff / 1000f);
            System.out.printf("Jacobi Method for %dx%d matrix error: %f\n", n, n, A.mult(j).minus(b).normF());
            time = System.currentTimeMillis();
            SimpleMatrix gS = myMain.gaussSeidel(A, b);
            diff = System.currentTimeMillis() - time;
            System.out.printf("Gauss-Seidel Method for %dx%d matrix time: %f seconds\n", n, n, diff / 1000f);
            System.out.printf("Gauss-Seidel Method for %dx%d matrix error: %f\n", n, n, A.mult(gS).minus(b).normF());
            time = System.currentTimeMillis();
            SimpleMatrix s = myMain.sor(A, b, w);
            diff = System.currentTimeMillis() - time;
            System.out.printf("SOR Method for %dx%d matrix time: %f seconds\n", n, n, diff / 1000f);
            System.out.printf("SOR Method for %dx%d matrix error: %f\n", n, n, A.mult(s).minus(b).normF());
        }
    }

    public SimpleMatrix[] lu(SimpleMatrix A, boolean partialPivoting) {
        int n = A.numCols();
        SimpleMatrix[] lArray = new SimpleMatrix[n - 1];
        SimpleMatrix[] pArray = new SimpleMatrix[n - 1];
        for (int i = 0; i < n - 1; i++) {
            if (partialPivoting) {
                SimpleMatrix permMatt = SimpleMatrix.identity(n);
                int maxIdx = i;
                double maxVal = A.get(i, i);
                for (int j = i; j < n; j++) {
                    if (A.get(j, i) > maxVal) {
                        maxIdx = j;
                    }
                }
                permMatt.set(i, i, 0);
                permMatt.set(i, maxIdx, 1);
                permMatt.set(maxIdx, maxIdx, 0);
                permMatt.set(maxIdx, i, 1);
                A = permMatt.mult(A);
                pArray[i] = permMatt;
            }
            SimpleMatrix l = SimpleMatrix.identity(n);
            SimpleMatrix m = A.cols(i, i + 1);

            double p = m.get(i, 0);
            for (int j = 0; j < n; j++) {
                if (j < i) {
                    l.set(j, i, 0);
                }
                if (j == i) {
                    l.set(j, i, 1);
                }
                if (j > i) {
                    double a = m.get(j, 0);
                    double a1 = (a * -1) / p;
                    l.set(j, i, a1);
                }
            }
            A = l.mult(A);
            lArray[i] = l;
        }
        SimpleMatrix lr = SimpleMatrix.identity(n);
        if (partialPivoting) {
            SimpleMatrix[] nlArray = new SimpleMatrix[n - 1];
            for (int i = 0; i < n - 1; i++) {
                SimpleMatrix id = lArray[i];
                for (int j = i + 1; j < n - 1; j++) {
                    id = pArray[j].mult(id).mult(pArray[j].invert());
                }
                nlArray[i] = id;
            }
            for (int i = 0; i < n - 1; i++) {
                lr = lr.mult(nlArray[i].invert());
            }
        } else {
            for (int i = 0; i < n - 1; i++) {
                lr = lr.mult(lArray[i].invert());
            }
        }
        if(partialPivoting){
            SimpleMatrix pm = SimpleMatrix.identity(n);
            for (int i = 0; i < n - 1; i++) {
                pm = pm.mult(pArray[n - 2 - i]);
            }
            lr = pm.invert().mult(lr);
        }
        SimpleMatrix[] result = new SimpleMatrix[2];
        result[0] = lr;
        result[1] = A;
        return result;
    }
    public SimpleMatrix[] luScalar(SimpleMatrix A)
    {
        // Doolittle Method
        int n = A.numCols();
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        for (int i = 0; i < n; i++) {
            // Calc U
            for (int k = i; k < n; k++)
            {
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[i][j] * U[j][k]);
                U[i][k] = A.get(i, k) - sum;
            }
            // Calc L
            for (int k = i; k < n; k++)
            {
                if (i == k)
                    L[i][i] = 1; // Doolittle
                else
                {
                    int sum = 0;
                    for (int j = 0; j < i; j++)
                        sum += (L[k][j] * U[j][i]);
                    L[k][i] = (A.get(k, i) - sum) / U[i][i];
                }
            }
        }
        SimpleMatrix sL = SimpleMatrix.identity(n);
        SimpleMatrix sU = SimpleMatrix.identity(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++){
                sL.set(i, j, L[i][j]);
                sU.set(i, j, U[i][j]);
            }
        }
        SimpleMatrix[] result = new SimpleMatrix[2];
        result[0] = sL;
        result[1] = sU;
        return result;
    }

    static int METHOD_LU_MATRIX = 2;
    static int METHOD_LU_SCALAR = 3;
    static int METHOD_LU_LIB = 4;

    public static void luTest(int method){
        Main myMain = new Main();
        int[] dims = new int[]{64, 256, 1024, 4096};
        long bef = -1;
        for (int n : dims) {
            SimpleMatrix A = myMain.createTriDiag(n);
            long time = System.currentTimeMillis();
            if(method == METHOD_LU_LIB){
                DecompositionFactory_DDRM.lu().decompose(A.getDDRM());
            }else if(method == METHOD_LU_MATRIX){
                SimpleMatrix[] lu = myMain.lu(A, false);
            }else if(method == METHOD_LU_SCALAR){
                SimpleMatrix[] lu = myMain.luScalar(A);
            }
            long diff = System.currentTimeMillis() - time;
            System.out.printf("LU Decomposition for %dx%d matrix time: %f seconds\n", n, n, diff / 1000f);
            if(bef != -1)
                System.out.printf("Change Scale: %f\n", ((float) diff / bef));
            bef = diff;
        }
    }

    public static void pivotTest(){
        Main myMain = new Main();
        SimpleMatrix A = myMain.createTriDiag(64);
        SimpleMatrix x = SimpleMatrix.diag(1000, 2, 7);
        x.set(0,1  , 500);
        x.set(0,2  , 100);
        x.set(1,0  , 1);
        x.set(1,2  , 0);
        x.set(2,0  , 1);
        x.set(2,1  , 5);
//        A = x;
        SimpleMatrix[] r1 = myMain.lu(A, false);
        SimpleMatrix[] r2 = myMain.lu(A, true);
        System.out.printf("Without pivot residual: %f\n", r1[0].mult(r1[1]).minus(A).elementPower(2).elementSum());
        System.out.printf("With pivot residual: %f\n", r2[0].mult(r2[1]).minus(A).elementPower(2).elementSum());
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Iterative Methods and LU Project:");
        System.out.println("1 - Iterative Methods Benchmark");
        System.out.println("2 - LU Benchmark - matrix project implementation");
        System.out.println("3 - LU Benchmark - scalar project implementation");
        System.out.println("4 - LU Benchmark - library implementation");
        System.out.println("5 - LU Benchmark - partial pivot vs simple method");
        System.out.print("Enter the task number to execute:");

        int i = scanner.nextInt();
        switch (i){
            case 1:
                iterativeMethods();
                break;
            case 2:
            case 3:
            case 4:
                luTest(i);
                break;
            case 5:
                pivotTest();
            default:
                break;
        }

    }
}
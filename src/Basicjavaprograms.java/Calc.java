import java.util.Scanner;
class Calc
{
    void add (double a, double b)
    {
        System.out.println("Addition is"+(a+b));
    }
    void sub (double a, double b)
    {
        System.out.println("Subtraction is"+(a-b));
    }
    void mul (double a, double b)
    {
        System.out.println("Multiplication is"+(a*b));
    }
    void div (double a ,double b)
    {
        System.out.println("Division is"+(a/b));
    }

    public static void main(String[] args) {
    try ( Scanner C =new Scanner(System.in))
      {
        Calc calc=new Calc();
        System.out.println("--|CALCULATOR|");
        System.out.println("Enter your first number:");
        double n1=C.nextDouble();
        System.out.println("Enter your second number:");
        double n2=C.nextDouble();
        System.out.println("1.ADDITION\n2.SUBTRACTION\n3.MULTIPLICATION\n4.DIVISION\n");
        System.out.println("Enter your choice");
        int choice=C.nextInt();
        switch (choice) {
            case 1 -> calc.add(n1, n2);
            case 2 -> calc.sub(n1, n2);
            case 3 -> calc.mul(n1, n2);
            case 4 -> calc.div(n1, n2);
            default -> System.out.println("INVALID CHOICE");
        }
        C.close();
    }
    } 
}
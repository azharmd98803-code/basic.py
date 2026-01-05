class Sum
{
    int a,b,c;

     Sum(int x,int y)
    {
    a=x;
    b=y;
    }

    Sum(int x,int y, int z) 
    {
    a=x;
    b=y;
    c=z;
    }
    void display()
    {
        System.out.println("the sum of numbers is :" +(a+b+c));
    }
    public static void main(String[] args) 
    {
        Sum s1=new Sum(10, 20);
        Sum s2=new Sum(10,20,30);
        s1.display();
        s2.display();
    }
}
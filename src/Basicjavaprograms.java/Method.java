class Methodover
{
void sum(int a ,int b )
{
System.out.println("sum of 2 numbers is:"+(a+b));
}
void sum(int a,int b,int c)
{
System.out.println("sum of 3 numbers is:"+(a+b+c));
}
public static void main(String args[])
{
Methodover mtd=new Methodover();
mtd.sum(10,20,30);
mtd.sum(10,20);
}}

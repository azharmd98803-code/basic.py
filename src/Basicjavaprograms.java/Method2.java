 class Methodover2
{
void sum(int a ,int b )
{
System.out.println("sum of 2 numbers is:"+(a+b));
}
void sum(float a,float b)
{
System.out.println("sum of 2 numbers is:"+(a+b));
}
public static void main(String args[])
{
Methodover2 mtd2=new Methodover2();
mtd2.sum(10,20);
mtd2.sum(10.5f,20.9f);
}}
/*[9/28, 2:25 PM] azhar: class Pattern
{
public static void main(String args[])
{
for (int i=1;i<=4;i++)
{
for (int j=1;j<=i;j++)
{
System.out.print("*");
}
System.out.print("\n");
}}}*/

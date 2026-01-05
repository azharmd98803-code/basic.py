class Vehicle
{
    int speed=60;
    void msg()
    {
        System.out.println("Vehicle is running");
    }
}
class Bike extends Vehicle  
{
    int speed=100;
    void msg()
    {
        System.out.println("Bike is running");
    }
    void display()
    {
        System.out.println("bike speed is"+speed);
        System.out.println("vehicle speed is"+super.speed);
        msg();
        super.msg();
    }   
    public static void main(String[] args) {
        Bike b=new Bike();
        b.display();
    }
}
    
class Prime {
    public static void main(String args[]) {
        int count = 0;
        int n = Integer.parseInt(args[0]); // Parse input argument to integer

        // Loop to count the number of divisors
        for (int i = 1; i <= n; i++) {
            if (n % i == 0) {
                count++;    
            }
        }

        // Check if the number is prime
        if (count == 2) {
            System.out.println(n + " is a prime number");
        } else {
            System.out.println(n + " is not a prime number");
        }
    }
}
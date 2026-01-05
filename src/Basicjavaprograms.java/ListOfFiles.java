import java.io.*;
public class ListOfFiles
{
    public static void ListOfFiles1(File dirpath)
    {
        File[] filelist=dirpath.listFiles();
        for(File file:filelist)
        {
            if(file.isFile())
            {
                System.out.println(file.getName());
            }
            else
            {
                ListOfFiles1(file);
            }
        }
    }
    public static void main(String[] args) throws IOException
        {
            File file=new File(".");
            ListOfFiles1(file);
    }  
}

import java.awt.*;  
public class ScrollbarExample {  
public static void main (String[] args) {  
Frame f = new Frame("Scrollbar Example");  
Scrollbar sbv = new Scrollbar(); 
sbv.setBounds (100, 100, 40, 120);  
Scrollbar sbh = new Scrollbar(0);  
sbh.setBounds (180, 100, 140, 40); 
f.add(sbv);  
f.add(sbh); 
f.setSize(400,400);  
f.setLayout(null);  
f.setVisible(true);  
}  
}
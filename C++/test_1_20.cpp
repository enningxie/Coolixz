#include <iostream>
#include "Sales_item.h"
using namespace std;
int main(){
    Sales_item input;
    while(cin >> input){
        cout << "output: " << input;
    }
    cout << endl;
    return 0;
}

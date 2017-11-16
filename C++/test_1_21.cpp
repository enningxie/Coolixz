#include <iostream>
#include "Sales_item.h"
using namespace std;
int main(){
    Sales_item input, sum;
    while(cin >> input){
        sum += input;
    }
    cout << sum << endl;
    return 0;
}

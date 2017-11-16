#include <iostream>
using namespace std;
int main(){
    int sum = 0;
    for(int i = 10; i >= 0; --i){
        sum += i;
    }
    cout << sum << endl;
    return 0;
}

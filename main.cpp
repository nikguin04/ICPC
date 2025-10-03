#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* args[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int x_s,y_s,x_t,y_t,x_p,y_p; // Standing, 
    if (!(cin >> x_s >> y_s >> x_t >> y_t >> x_p >> y_p)) return 0;

    cout << x_s << ' ' << y_s << '\n' << x_t << ' ' << y_t << '\n';
    
    return 0;
}
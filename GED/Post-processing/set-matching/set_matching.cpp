#pragma GCC optimize(2)
#include<bits/stdc++.h>
#define N 305
#define K 300000
#define K_max 100
#define rand_time 3
#define LL long long
#define inf 2147483647777777
#define INF 2147483647214748364
#define LD long double
#define eps 1e-1
#define max_dif 1000000
#define dif 10
#define s_id set<int>::iterator
using namespace std;
int n,m,ans,bound,mapping[N][N],f[N],a[N],b[N];LL cost[N][N],c[N][N],c_tmp[N][N];vector<int> A[N],B[N];bool cut[N][N];
LL dis[N<<1],slack[N];int vis[N<<1],vis2[N<<1];bool in[N];/*vector<int> miss;*/
int tot,tt,vx[N],lst[N],ok[N][N],hd,tl,q[N*N*N],tmp_matching[N];// for choosing a random vertex
bool use[K+5];int Ans[K+5][15];double running_time[K+5][15];
int ged[N];LL bm[K+5];
struct graph{int n,m,u[N*N],v[N*N];string labels[N];}GA,GB;bool con1[N][N],con2[N][N];
struct path{
    bool del[N][N],add[N][N],out[N];int mapping[N],r_mapping[N];
    int gen_path(){
        // memset(del,0,sizeof(del)),memset(add,0,sizeof(add));
        // memset(out,0,sizeof(out));
        for(int j=1;j<=GB.n;j++) out[j]=0;for(int j=1;j<=GA.n;j++) r_mapping[mapping[j]]=j;
        for(int j=GA.n+1;j<=GB.n;j++) out[mapping[j]]=1;
        int res=GB.n-GA.n;for(int j=1;j<=GA.n;j++) if(GA.labels[j]!=GB.labels[mapping[j]]) res++;
        for(int j=1;j<=GB.m;j++) if(out[GB.u[j]]||out[GB.v[j]]) res++/*,del[GB.u[j]][GB.v[j]]=1*/;
        else if(!con1[r_mapping[GB.u[j]]][r_mapping[GB.v[j]]]) res++;
        for(int j=1;j<=GA.m;j++) if(!con2[mapping[GA.u[j]]][mapping[GA.v[j]]]) res++;
        // for(int j=1;j<=GA.n;j++) for(int k=j+1;k<=GA.n;k++) if(con1[j][k]^con2[mapping[j]][mapping[k]]) res++/*,del[j][k]=con1[j][k],add[mapping[j]][mapping[k]]=con2[mapping[j]][mapping[k]]*/;
        return res;
    }
}gt,pred/*,pred[K_max+5]*/;
int MCS[K+5],gr[K+5],num,acc;double E,recall,prec,f1,precision;LL e;double recall_=0,prec_=0,f1_=0,precision_=0;
double total_time[5];
int pre_mapping[N];
struct Graph{
    vector<pair<int,int> > I,O;// Matching M must contain I, and must not contain O
    int best_matching[N],second_matching[N],match[N],V;LL bm,sm;
    LL h[N<<1];// result of KM algorithm, for setting the edges' weight non-negative
    inline void print(){
        // cout/*<<"matching result: "*/<<bm<<' '/*<<"matching matrix:\n"*/;
        // for(int i=1;i<=n;cout<<'\n',i++) for(int j=1;j<=m;j++) if(best_matching[i]==j) cout<<1<<' ';else cout<<0<<' ';cout<<'\n';
    } // print the best matching
    inline void spfa(){
        int S=0;for(int i=1;i<=n;i++) h[i]=-inf,(!vis[i])&&(S=i,0);if(!S) return;h[S]=0;hd=1,tl=1;q[1]=S;
        while(hd<=tl){
            int x=q[hd++],y=best_matching[x];in[x]=0;
            for(int i=1;i<=n;i++) if(!vis[i]&&ok[i][y]!=tt&&h[i]<h[x]+cost[i][y]-cost[x][y])
            {h[i]=h[x]+cost[i][y]-cost[x][y];if(!in[i]) in[i]=1,q[++tl]=i;}
        }
    }
    inline void get_h(){// get the dual variable h
        tt++;for(int i=0;i<O.size();i++) ok[O[i].first][O[i].second]=tt;spfa();
        for(int i=1;i<=n;i++) h[best_matching[i]+n]=cost[i][best_matching[i]]-h[i];
    }
    inline void dijkstra(int S){
        int x=best_matching[S]+n;
        for(int i=1;i<=n+m;i++) dis[i]=INF;dis[S]=0,vis[S]=1;while(1){
            if(S<=n){for(int i=1;i<=m;i++) if(i!=best_matching[S]&&dis[S]+c[S][i]<dis[i+n]&&!vis[i+n]&&c[S][i]!=inf) dis[i+n]=dis[S]+c[S][i],lst[i]=S;}
            else{if(match[S-n]){dis[match[S-n]]=dis[S],S=match[S-n],vis[S]=1;continue;}} 
            S=0;for(int i=n+1;i<=n+m;i++) if(!vis[i]&&dis[i]!=INF) if(!S||dis[i]<dis[S]) S=i;if(!S) break;vis[S]=1;
            if(S==x) break;
        }
    }
    inline int get_lower_bound(){
        int res=abs(GA.n-GB.n);for(auto i:I) if(GA.labels[i.first]!=GB.labels[i.second]) res++;int m1=GA.m,m2=GB.m;
        for(int i=0;i<I.size();i++) for(int j=i+1;j<I.size();j++) if(con1[I[i].first][I[j].first]^con2[I[i].second][I[j].second]) res++;
        for(int i=0;i<I.size();i++) for(int j=i+1;j<I.size();j++) if(con1[I[i].first][I[j].first]) m1--;
        for(int i=0;i<I.size();i++) for(int j=i+1;j<I.size();j++) if(con2[I[i].second][I[j].second]) m2--;
        for(int i=1;i<=GA.n;i++) vis[i]=0;for(auto i:I) vis[i.first]=1;
        for(int i=1;i<=GB.n;i++) vis2[i]=0;for(auto i:I) vis2[i.second]=1;
        for(int i=0;i<I.size();i++){
            int c1=0,c2=0;
            for(int j=1;j<=GA.n;j++) if(!vis[j]&&con1[I[i].first][j]) c1++;
            for(int j=1;j<=GB.n;j++) if(!vis2[j]&&con2[I[i].second][j]) c2++;
            res+=abs(c2-c1),m1-=c1,m2-=c2;
        }
        res+=abs(m1-m2);for(int i=1;i<=GA.n;i++) vis[i]=0;for(int i=1;i<=GB.n;i++) vis2[i]=0;return res;
    }
    inline void get_ans_sec(){
        for(int j=1;j<=n;j++) pred.mapping[j]=second_matching[j];int tmp=pred.gen_path();ans=min(ans,tmp);bound=min(bound,tmp);
        int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][second_matching[j]]) res++;
        if(ans==tmp) precision_=1.0*res/GA.n;
        if(ans==tmp){for(int j=1;j<=n;j++) pre_mapping[j]=second_matching[j];}
    }
    inline void get_ans(){
        for(int j=1;j<=n;j++) pred.mapping[j]=best_matching[j];int tmp=pred.gen_path();ans=min(ans,tmp);bound=min(bound,tmp);
        int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][best_matching[j]]) res++;
        if(ans==tmp) precision_=1.0*res/GA.n;
        if(ans==tmp){for(int j=1;j<=n;j++) pre_mapping[j]=best_matching[j];}
    }
    inline void get_second(){
        clock_t start_time=clock();
        memset(second_matching,0,sizeof(second_matching)),memset(match,0,sizeof(match)),memset(vis,0,sizeof(vis));
        for(int i=0;i<I.size();i++) vis[I[i].first]=vis[I[i].second+n]=2;
        get_h();for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) if(best_matching[i]==j) c[i][j]=0,match[j]=i;else c[i][j]=h[i]+h[j+n]-cost[i][j];
        for(int i=0;i<O.size();i++) c[O[i].first][O[i].second]=inf;
        clock_t end_time=clock();
        total_time[0]+=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
        tot=0;for(int i=1;i<=n;i++) if(!vis[i]) vx[++tot]=i;if(!tot){sm=-INF;return;} 
        for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) cut[i][j]=0;for(int i=1;i<=n+m;i++) vis2[i]=vis[i];
        V=0,sm=-INF;
        for(int i=1;i<=tot;i++){
            start_time=clock();
            int S=vx[i],x=best_matching[S],sa=a[S],sb=b[x];if(cut[sa][sb]) continue;cut[sa][sb]=1;
            for(int ii=1;ii<=n+m;ii++) vis[ii]=vis2[ii];
            for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++){int k=A[sa][ii],j=B[sb][jj];if(best_matching[k]^j) c_tmp[k][j]=c[k][j],c[k][j]=inf;}
            dijkstra(S);
            end_time=clock();
            total_time[1]+=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
            start_time=clock();
            if(dis[x+n]<INF){
                for(int j=1;j<=n;j++) tmp_matching[j]=second_matching[j];LL tmp_sm=sm,tmp_V=V;
                sm=bm-dis[x+n],V=S;for(int j=1;j<=n;j++) second_matching[j]=best_matching[j];
                while(1){second_matching[lst[x]]=x;if(lst[x]==S) break;x=best_matching[lst[x]];} // find the cycle and get the second-best-matching
                get_ans_sec();
                if(sm<=tmp_sm){for(int j=1;j<=n;j++) second_matching[j]=tmp_matching[j];sm=tmp_sm,V=tmp_V;}
                // else get_ans_sec();
            }
            end_time=clock();
            for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++){int k=A[sa][ii],j=B[sb][jj];if(best_matching[k]^j) c[k][j]=c_tmp[k][j];}
            total_time[2]+=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
        }
        
    }
    inline bool check(int x){
        vis[x+n]=1;if(match[x]!=-1){q[++tl]=match[x],vis[match[x]]=1;return 0;}
        while(x!=-1) match[x]=lst[x],swap(x,best_matching[lst[x]]);return 1;
    }
    inline void bfs(int S){
        hd=1,tl=1;q[1]=S,vis[S]=1;while(1){
            while(hd<=tl){
                int x=q[hd++];for(int i=1;i<=n;i++) if(!vis[i+n]){
                    LL d=h[x]+h[i+n]-cost[x][i];if(slack[i]>=d)
                    {lst[i]=x;if(d) slack[i]=d;else if(check(i)) return;}
                }
            }
            LL res=INF;for(int i=1;i<=n;i++) if(!vis[i+n]) res=min(res,slack[i]);
            for(int i=1;i<=n;i++){if(vis[i]) h[i]-=res;if(vis[i+n]) h[i+n]+=res;else slack[i]-=res;}
            for(int i=1;i<=n;i++) if(!vis[i+n]&&!slack[i]&&check(i)) return;
        }
    }
    inline void KM(){
        for(int i=1;i<=n;i++){match[i]=best_matching[i]=-1,h[i]=h[i+n]=0;for(int j=1;j<=n;j++) h[i]=max(h[i],cost[i][j]);}
        for(int i=1;i<=n;i++){for(int j=1;j<=n;j++) slack[j]=inf;memset(vis,0,sizeof(vis)),bfs(i);}
        bm=0;for(int i=1;i<=n;i++) bm+=cost[i][best_matching[i]];
    }
}G[K_max+5];// K subgraphs
inline bool cmp(int x,int y){return G[x].bm>G[y].bm;}
inline void remove_edge(){
    int mm=0;for(int i=1;i<=GA.m;i++) if(!con1[GA.u[i]][GA.v[i]]) con1[GA.v[i]][GA.u[i]]=con1[GA.u[i]][GA.v[i]]=1,mm++,GA.u[mm]=GA.u[i],GA.v[mm]=GA.v[i];GA.m=mm;
    mm=0;for(int i=1;i<=GB.m;i++) if(!con2[GB.u[i]][GB.v[i]]) con2[GB.v[i]][GB.u[i]]=con2[GB.u[i]][GB.v[i]]=1,mm++,GB.u[mm]=GB.u[i],GB.v[mm]=GB.v[i];GB.m=mm;
    for(int i=1;i<=GA.n;i++) for(int j=1;j<=GA.n;j++) con1[i][j]=0;for(int i=1;i<=GB.n;i++) for(int j=1;j<=GB.n;j++) con2[i][j]=0;
}
inline void read(){
    cin>>n>>m;for(int i=1;i<=n;i++){double x;for(int j=1;j<=m;j++) cin>>x,cost[i][j]=(LL)x;}
    for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) cin>>mapping[i][j];
    GA.n=n;for(int i=1;i<=n;i++) cin>>GA.labels[i];cin>>GA.m;for(int i=1;i<=GA.m;i++) cin>>GA.u[i]>>GA.v[i],GA.u[i]++,GA.v[i]++;
    GB.n=m;for(int i=1;i<=m;i++) cin>>GB.labels[i];cin>>GB.m;for(int i=1;i<=GB.m;i++) cin>>GB.u[i]>>GB.v[i],GB.u[i]++,GB.v[i]++;
    // remove_edge();  note: this step is for IMDB dataset, where an undirected edge (u,v) appears twice as (u,v),(v,u) in the input graph
}
inline int gf(int x){return f[x]==x?x:f[x]=gf(f[x]);}
vector<int> NB[N<<1],NB2[N<<1];
int app_matching[N];bool increment[N][N];
// inline void update(){
//     for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) increment[i][j]=0;
//     for(int i=1;i<=GA.n;i++) cin>>app_matching[i],increment[a[i]][b[app_matching[i]]]=1;
//     for(int i=1;i<=GA.n;i++) for(int j=1;j<=GB.n;j++) if(increment[a[i]][b[j]]) cost[i][j]=(LL)(cost[i][j]*1.1);
// }
inline void init(){
    clock_t start_time=clock();
    for(int i=1;i<=GA.n+GB.n;i++) NB[i].clear(),NB2[i].clear();
    for(int i=1;i<=GA.m;i++) NB[GA.u[i]].push_back(GA.v[i]),NB[GA.v[i]].push_back(GA.u[i]);
    for(int i=1;i<=GB.m;i++) NB[GB.u[i]+n].push_back(GB.v[i]),NB[GB.v[i]+n].push_back(GB.u[i]);
    for(int i=1;i<=GA.n+GB.n;i++){NB2[i]=NB[i];if(i<=GA.n) NB2[i].push_back(i);else NB2[i].push_back(i-GA.n);sort(NB[i].begin(),NB[i].end()),sort(NB2[i].begin(),NB2[i].end());}
    memset(vis,0,sizeof(vis));G[1].I.clear(),G[1].O.clear();
    for(int i=n+1;i<=m;i++){for(int j=1;j<=m;j++) cost[i][j]=0;} n=m;
    // for(int i=1;i<=GA.n;i++) for(int j=1;j<=GB.n;j++) if(GA.labels[i]!=GB.labels[j]) cost[i][j]=0;
    //merge the set
    for(int i=1;i<=GA.n;i++) f[i]=i;for(int i=GA.n+1;i<=GB.n;i++) f[i]=GA.n+1;// row
    // for(int i=1;i<=n;i++){double c_max=0,c_min=inf;for(int j=1;j<=n;j++) c_max=max(c_max,cost[i][j]),c_min=min(c_min,cost[i][j]);if(c_max-c_min<max_dif){case1[i]=1;for(int j=1;j<=n;j++) cost[i][j]=c_min;}}
    // for(int i=1;i<=n;i++) if(case1[i]){for(int j=i+1;j<=n;j++) if(case1[j]) f[gf(j)]=gf(i);break;} //case 1: c_max-c_min<max_dif
    // for(int i=1;i<=n;i++) for(int j=i+1;j<=n;j++) if(gf(i)^gf(j)){bool flg=1;for(int k=1;k<=n;k++) if(abs(cost[i][k]-cost[j][k])>dif){flg=0;break;} if(flg){for(int k=1;k<=n;k++) cost[j][k]=cost[i][k];f[gf(j)]=gf(i);}} //case 2: cost[i][k]=cost[j][k]
    for(int i=1;i<=GA.n;i++) for(int j=i+1;j<=GA.n;j++) if(gf(i)^gf(j)) if((NB[i]==NB[j]||NB2[i]==NB2[j])&&GA.labels[i]==GA.labels[j]) f[gf(j)]=gf(i);
    tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,A[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i){a[j]=tot,A[tot].push_back(j);for(int k=1;k<=n;k++) cost[j][k]=cost[i][k];}}
    // for(int i=1;i<=tot;i++) cout<<A[i].size()<<' ';cout<<'\n';
    for(int i=1;i<=n;i++) f[i]=i;// column
    // for(int i=1;i<=n;i++){double c_max=0,c_min=inf;for(int j=1;j<=n;j++) c_max=max(c_max,cost[j][i]),c_min=min(c_min,cost[j][i]);if(c_max-c_min<max_dif){case1[i]=1;for(int j=1;j<=n;j++) cost[j][i]=c_min;}}
    // for(int i=1;i<=n;i++) if(case1[i]){for(int j=i+1;j<=n;j++) if(case1[j]) f[gf(j)]=gf(i);break;} //case 1: c_max-c_min<max_dif
    for(int i=1;i<=GB.n;i++) for(int j=i+1;j<=GB.n;j++) if(gf(i)^gf(j)) if((NB[i+GA.n]==NB[j+GA.n]||NB2[i+GA.n]==NB2[j+GA.n])&&GB.labels[i]==GB.labels[j]) f[gf(j)]=gf(i);
    tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,B[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i){b[j]=tot,B[tot].push_back(j);for(int k=1;k<=n;k++) cost[k][j]=cost[k][i];}}
    // update();
    tot=0,G[1].KM();
    clock_t end_time=clock();
    // total_time[2]+=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
}
int KK;
inline void AKM(int tc){
    clock_t start_time=clock();
    bool flg=1;G[1].print();G[1].get_ans(),G[1].get_second();// Get the second-best matching
    for(int i=0;i<=10;i++) Ans[tc][i]=888888,running_time[tc][i]=0;
    Ans[tc][0]=ans;
    clock_t end_time=clock();
    running_time[tc][0]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
    int tmp=ans;
    for(int k=2;k<=K_max;k++){
        int id=1;for(int i=2;i<k;i++) if(G[i].sm>G[id].sm) id=i;// find the maximum second_best_matching
        if(G[id].sm==-INF){KK=k-1;/*cout<<"Only "<<k-1<<" matchings exist"<<'\n'<<'\n';*/flg=0;for(int i=k;i<=K_max;i++) G[i].bm=-INF;/*for(int i=1;i<=n;cout<<'\n',i++) for(int j=1;j<=n;j++) cout<<cost[i][j]<<' ';*/break;}
        int S=G[id].V,x=G[id].best_matching[S],sa=a[S],sb=b[x];G[k].I=G[id].I,G[k].O=G[id].O,G[k].O.push_back(make_pair(S,x));
        for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++)
        {int i=A[sa][ii],j=B[sb][jj];if(G[id].best_matching[i]==j) G[id].I.push_back(make_pair(i,j));else G[k].O.push_back(make_pair(i,j));}
        memcpy(G[k].best_matching,G[id].second_matching,sizeof(G[id].second_matching)),G[k].bm=G[id].sm,G[k].print();
        G[id].get_second(),G[k].get_second();// update I&O and second_best_matching
        // int tmp=G[id].get_lower_bound();
        // if(tmp-bound>=-bound*eps) G[id].sm=-INF;
        // tmp=G[k].get_lower_bound();
        // if(tmp-bound>=-bound*eps) G[k].sm=-INF;
        if(G[id].get_lower_bound()>=bound) G[id].sm=-INF;if(G[k].get_lower_bound()>=bound) G[k].sm=-INF;
        // if(k%20==0){if(ans==tmp){KK=k-1;flg=0;break;} tmp=ans;}
        // if(k%10==0){
        //     end_time=clock();
        //     Ans[tc][k/10]=ans;
        //     running_time[tc][k/10]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
        // }
    }
    end_time=clock();
    // for(int i=1;i<=10;i++){
    //     if(i*10>KK) running_time[tc][i]=max(running_time[tc][i],(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC)),Ans[tc][i]=ans;
    // }
    // if(flg) cout<<K<<"-best matchings are found"<<'\n'<<'\n';
}
int opt[K+5],sum_K;
inline void solve(int tc){
    KK=K_max;ans=888888,bound=/*opt[tc]*/888888;
    // cout<<"Test data "<<tc<<":\n";
    precision_=0;
    read();// read the data

    // if(tc!=11249&&tc!=11300&&tc!=11203) return;

    clock_t start_time=clock();

    // if(!use[tc]) return;
    init();// Get the matching matrix and the best matching
    for(int j=1;j<=GA.m;j++) con1[GA.u[j]][GA.v[j]]=con1[GA.v[j]][GA.u[j]]=1;for(int j=1;j<=GB.m;j++) con2[GB.u[j]][GB.v[j]]=con2[GB.v[j]][GB.u[j]]=1;
    AKM(tc);// Algorithm of K-best Matching
    // for(int i=1;i<=K;i++) bm[(i-1)/100+1]+=G[i].bm,cout<<G[i].bm<<' ';cout<<'\n';
    // for(int j=1;j<=GA.n;j++) gt.mapping[j]=mapping[j];gt.gen_path();
    // bool flg=0;
    // for(int i=1;i<=KK;i++){
    //     for(int j=1;j<=n;j++) pred[i].mapping[j]=G[i].best_matching[j];int tmp=pred[i].gen_path();ans=min(ans,tmp);
    //     int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][G[i].best_matching[j]]) res++;precision_=max(precision_,1.0*res/GA.n);
    //     // int intersec=0;
    //     // for(int j=1;j<=GA.n;j++) if(GA.labels[j]!=GB.labels[gt.mapping[j]]&&pred[i].mapping[j]==gt.mapping[j]) intersec++;
    //     // for(int j=1;j<=GB.n;j++) if(pred[i].out[j]&&gt.out[j]) intersec++;
    //     // for(int j=1;j<=GA.n;j++) for(int k=j+1;k<=GA.n;k++) if(pred[i].del[j][k]&&gt.del[j][k]) intersec++;
    //     // for(int j=1;j<=GB.n;j++) for(int k=j+1;k<=GB.n;k++) if(pred[i].add[j][k]&&gt.add[j][k]) intersec++;
    //     // if(gr[tc]){recall_=max(recall_,1.0*intersec/gr[tc]),prec_=max(prec_,1.0*intersec/tmp);if(intersec) f1_=max(f1_,2.0/(1.0*gr[tc]/intersec+1.0*tmp/intersec));}
    //     // if(ans==gr[tc]){flg=1,cout<<i<<'\n';break;}
    // }
    // if(!flg) cout<<K+1<<'\n';

    cout<<"GED = "<<ans<<'\n';
    cout<<"Matching = ";for(int i=1;i<=GA.n;i++) if(pre_mapping[i]<=GB.n) cout<<"("<<i-1<<","<<pre_mapping[i]-1<<") ";cout<<'\n';

    for(int j=1;j<=GA.m;j++) con1[GA.u[j]][GA.v[j]]=con1[GA.v[j]][GA.u[j]]=0;for(int j=1;j<=GB.m;j++) con2[GB.u[j]][GB.v[j]]=con2[GB.v[j]][GB.u[j]]=0;
    // cout<<ans<<' ';
    // for(int i=0;i<=10;i++) cout<<Ans[tc][i]<<' ';
    // for(int i=0;i<=10;i++) cout<<running_time[tc][i]<<' ';
    // cout<<'\n';
    if(ans!=gr[tc]) E+=1.0*(ans-gr[tc])/gr[tc],e+=ans-gr[tc];//MAE
    else acc++,precision_=1;
    precision+=precision_,num++;
    // cout<<precision_<<' ';
    clock_t end_time=clock();
    // cout<<(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC)<<'\n';

    // if(tc%100==0) cerr<<tc<<'\n';
    sum_K+=KK;
    // if(gr[tc]) recall+=recall_,prec+=prec_,f1_+=f1,num++;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    // freopen("use.txt","r",stdin);
    // int n_use;cin>>n_use;while(n_use--){int x;cin>>x;for(int i=(x-1)*20+1;i<=x*20;i++) use[i]=1;}
    // freopen("ground_truth_GED_AIDS.txt","r",stdin);
    int tc=1;
    // cin>>tc;
    // tc=10;
    // for(int i=1;i<=tc;i++) cin>>gr[i];
    // freopen("./App-Bmao/App_Bmao_IMDBMulti.txt","r",stdin);
    // for(int i=1;i<=tc;i++){double x;cin>>opt[i]>>x;}
    // freopen("val_data_GED_AIDS.txt","r",stdin);
    // freopen("case_study_set_matching_1000.txt","w",stdout);
    // freopen("set_matching_lb_IMDBMulti_70.txt","w",stdout);

    clock_t start_time=clock();

    srand(time(NULL));
    // int tc=1;
    // cin>>tc;
    // tc=10;
    for(int i=1;i<=tc;i++) solve(i);
    
    // for(int i=1;i<=K/100;i++) printf("%.3f ",1.0*bm[i]/10000.0);

    // for(int i=1;i<=K/100;i++) printf("%.3f ",MCS[i]/1000.0);

    // cout<<1.0*e/tc<<' '<<E/tc<<'\n';

    // cout<<1.0*e/num<<' ';//MAE

    // cout<<prec/num<<' '<<recall/num<<' '<<f1/num<<' ';//prec,recall,f1

    // cout<<1.0*acc/num<<' ';//ACC

    // cout<<precision/num<<' ';//precision

    // cout<<1.0*sum_K/num<<' ';

    // cout<<total_time[0]<<' '<<total_time[1]<<' '<<total_time[2]<<'\n';

    // clock_t end_time=clock();

    // printf("%.4f\n",(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC));//time
    return 0;
}

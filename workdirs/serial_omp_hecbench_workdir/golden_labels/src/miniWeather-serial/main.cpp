















#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

const double pi        = 3.14159265358979323846264338327;   

const double grav      = 9.8;                               

const double cp        = 1004.;                             

const double cv        = 717.;                              

const double rd        = 287.;                              

const double p0        = 1.e5;                              

const double C0        = 27.5629410929725921310572974482;   

const double gamm      = 1.40027894002789400278940027894;   



const double xlen      = 2.e4;    

const double zlen      = 1.e4;    

const double hv_beta   = 0.25;     

const double cfl       = 1.50;    

const double max_speed = 450;        

const int hs        = 2;          

const int sten_size = 4;          




const int NUM_VARS = 4;           

const int ID_DENS  = 0;           

const int ID_UMOM  = 1;           

const int ID_WMOM  = 2;           

const int ID_RHOT  = 3;           

const int DIR_X = 1;              

const int DIR_Z = 2;              

const int DATA_SPEC_COLLISION       = 1;
const int DATA_SPEC_THERMAL         = 2;
const int DATA_SPEC_MOUNTAIN        = 3;
const int DATA_SPEC_TURBULENCE      = 4;
const int DATA_SPEC_DENSITY_CURRENT = 5;
const int DATA_SPEC_INJECTION       = 6;

const int nqpoints = 3;
double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };







double sim_time;              

double dt;                    

int    nx, nz;                

double dx, dz;                

int    nx_glob, nz_glob;      

int    i_beg, k_beg;          

int    nranks, myrank;        

int    left_rank, right_rank; 

int    masterproc;            

double data_spec_int;         

double *hy_dens_cell;         

double *hy_dens_theta_cell;   

double *hy_dens_int;          

double *hy_dens_theta_int;    

double *hy_pressure_int;      








double etime;                 

double output_counter;        



double *state;                

double *state_tmp;            

double *flux;                 

double *tend;                 

double *sendbuf_l;            

double *sendbuf_r;            

double *recvbuf_l;            

double *recvbuf_r;            

int    num_out = 0;           

int    direction_switch = 1;
double mass0, te0;            

double mass , te ;            




double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };




void   init                 ( int *argc , char ***argv );
void   finalize             ( );
void   injection            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   density_current      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   turbulence           ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   mountain_waves       ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   thermal              ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   collision            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   hydro_const_theta    ( double z                   , double &r , double &t );
void   hydro_const_bvfreq   ( double z , double bv_freq0 , double &r , double &t );
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   perform_timestep     ( double *state , double *state_tmp , double *flux , double *tend , double dt );
void   semi_discrete_step   ( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend );
void   compute_tendencies_x ( double *state , double *flux , double *tend );
void   compute_tendencies_z ( double *state , double *flux , double *tend );
void   set_halo_values_x    ( double *state );
void   set_halo_values_z    ( double *state );
void   reductions           ( double &mass , double &te );








int main(int argc, char **argv) {
  

  

  

  

  

  nx_glob = NX;      

  nz_glob = NZ;       

  sim_time = SIM_TIME;     

  data_spec_int = DATA_SPEC;  

  

  

  


  init( &argc , &argv );
{

  

  reductions(mass0,te0);

  

  

  

  auto c_start = clock();
  while (etime < sim_time) {
    

    if (etime + dt > sim_time) { dt = sim_time - etime; }
    

    perform_timestep(state,state_tmp,flux,tend,dt);
    

    etime = etime + dt;
  }
  auto c_end = clock();
  if (masterproc) {
     printf("CPU Time: %lf sec\n",( (double) (c_end-c_start) ) / CLOCKS_PER_SEC);
  }

  

  reductions(mass,te);
}

  printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
  printf( "d_te:   %le\n" , (te   - te0  )/te0   );

  finalize();
}
















void perform_timestep( double *state , double *state_tmp , double *flux , double *tend , double dt ) {
  if (direction_switch) {
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
  } else {
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}








void semi_discrete_step( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend ) {
  int i, k, ll, inds, indt;
  if        (dir == DIR_X) {
    

    set_halo_values_x(state_forcing);
    

    compute_tendencies_x(state_forcing,flux,tend);
  } else if (dir == DIR_Z) {
    

    set_halo_values_z(state_forcing);
    

    compute_tendencies_z(state_forcing,flux,tend);
  }

  

  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        indt = ll*nz*nx + k*nx + i;
        state_out[inds] = state_init[inds] + dt * tend[indt];
      }
    }
  }
}










void compute_tendencies_x( double *state , double *flux , double *tend ) {
  int    i,k,ll,s,inds,indf1,indf2,indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  

  hv_coef = -hv_beta * dx / (16*dt);
  

  for (k=0; k<nz; k++) {
    for (i=0; i<nx+1; i++) {
      

      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < sten_size; s++) {
          inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
          stencil[s] = state[inds];
        }
        

        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        

        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      

      r = vals[ID_DENS] + hy_dens_cell[k+hs];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
      p = C0*pow((r*t),gamm);

      

      flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  

  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        indt  = ll* nz   * nx    + k* nx    + i  ;
        indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
        indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
      }
    }
  }
}










void compute_tendencies_z( double *state , double *flux , double *tend ) {
  int    i,k,ll,s, inds, indf1, indf2, indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  

  hv_coef = -hv_beta * dz / (16*dt);
  

  for (k=0; k<nz+1; k++) {
    for (i=0; i<nx; i++) {
      

      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<sten_size; s++) {
          inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
          stencil[s] = state[inds];
        }
        

        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        

        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      

      r = vals[ID_DENS] + hy_dens_int[k];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
      p = C0*pow((r*t),gamm) - hy_pressure_int[k];
      

      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals[ID_DENS] = 0;
      }

      

      flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  

  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        indt  = ll* nz   * nx    + k* nx    + i  ;
        indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
        indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;
        if (ll == ID_WMOM) {
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
          tend[indt] = tend[indt] - state[inds]*grav;
        }
      }
    }
  }
}




void set_halo_values_x( double *state ) {
  int k, ll, ind_r, ind_u, ind_t, i, s, ierr;
  double z;
  MPI_Request req_r[2], req_s[2];

  

  ierr = MPI_Irecv(recvbuf_l,hs*nz*NUM_VARS,MPI_DOUBLE, left_rank,0,MPI_COMM_WORLD,&req_r[0]);
  ierr = MPI_Irecv(recvbuf_r,hs*nz*NUM_VARS,MPI_DOUBLE,right_rank,1,MPI_COMM_WORLD,&req_r[1]);

  

  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (s=0; s<hs; s++) {
        sendbuf_l[ll*nz*hs + k*hs + s] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+s];
        sendbuf_r[ll*nz*hs + k*hs + s] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+s];
      }
    }
  }


  

  ierr = MPI_Isend(sendbuf_l,hs*nz*NUM_VARS,MPI_DOUBLE, left_rank,1,MPI_COMM_WORLD,&req_s[0]);
  ierr = MPI_Isend(sendbuf_r,hs*nz*NUM_VARS,MPI_DOUBLE,right_rank,0,MPI_COMM_WORLD,&req_s[1]);

  

  ierr = MPI_Waitall(2,req_r,MPI_STATUSES_IGNORE);


  

  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (s=0; s<hs; s++) {
        state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + s      ] = recvbuf_l[ll*nz*hs + k*hs + s];
        state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+s] = recvbuf_r[ll*nz*hs + k*hs + s];
      }
    }
  }

  

  ierr = MPI_Waitall(2,req_s,MPI_STATUSES_IGNORE);

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      for (k=0; k<nz; k++) {
        for (i=0; i<hs; i++) {
          z = (k_beg + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            state[ind_u] = (state[ind_r]+hy_dens_cell[k+hs]) * 50.;
            state[ind_t] = (state[ind_r]+hy_dens_cell[k+hs]) * 298. - hy_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }

}






void set_halo_values_z( double *state ) {
  int          i, ll;
  const double mnt_width = xlen/8;
  double       x, xloc, mnt_deriv;
  for (ll=0; ll<NUM_VARS; ll++) {
    for (i=0; i<nx+2*hs; i++) {
      if (ll == ID_WMOM) {
        state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
        

        if (data_spec_int == DATA_SPEC_MOUNTAIN) {
          x = (i_beg+i-hs+0.5)*dx;
          if ( fabs(x-xlen/4) < mnt_width ) {
            xloc = (x-(xlen/4)) / mnt_width;
            

            mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
            

            state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (0)*(nx+2*hs) + i] = mnt_deriv*state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
            state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (1)*(nx+2*hs) + i] = mnt_deriv*state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
          }
        }
      } else {
        state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
      }
    }
  }
}


void init( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, ierr, inds, i_end;
  double x, z, r, u, w, t, hr, ht, nper;

  ierr = MPI_Init(argc,argv);

  

  dx = xlen / nx_glob;
  dz = zlen / nz_glob;

  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  nper = ( (double) nx_glob ) / nranks;
  i_beg = round( nper* (myrank)    );
  i_end = round( nper*((myrank)+1) )-1;
  nx = i_end - i_beg + 1;
  left_rank  = myrank - 1;
  if (left_rank == -1) left_rank = nranks-1;
  right_rank = myrank + 1;
  if (right_rank == nranks) right_rank = 0;


  

  

  

  

  


  

  k_beg = 0;
  nz = nz_glob;
  masterproc = (myrank == 0);

  

  state              = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  flux               = (double *) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(double) );
  tend               = (double *) malloc( nx*nz*NUM_VARS*sizeof(double) );
  hy_dens_cell       = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_theta_cell = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_int        = (double *) malloc( (nz+1)*sizeof(double) );
  hy_dens_theta_int  = (double *) malloc( (nz+1)*sizeof(double) );
  hy_pressure_int    = (double *) malloc( (nz+1)*sizeof(double) );
  sendbuf_l          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  sendbuf_r          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  recvbuf_l          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  recvbuf_r          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );

  

  dt = dmin(dx,dz) / max_speed * cfl;
  

  etime = 0.;
  output_counter = 0.;

  

  if (masterproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }
  

  ierr = MPI_Barrier(MPI_COMM_WORLD);

  

  

  

  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      

      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state[inds] = 0.;
      }
      

      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          

          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          

          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          

          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state_tmp[inds] = state[inds];
      }
    }
  }
  

  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      [k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      

      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      [k] = hy_dens_cell      [k] + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  

  for (k=0; k<nz+1; k++) {
    z = (k_beg + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr*ht;
    hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}










void injection( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}










void density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}








void turbulence( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  

  

  

  

}








void mountain_waves( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}










void thermal( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}










void collision( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}








void hydro_const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  

  const double exner0 = 1.;    

  double       p,exner,rt;
  

  t = theta0;                                  

  exner = exner0 - grav * z / (cp * theta0);   

  p = p0 * pow(exner,(cp/rd));                 

  rt = pow((p / C0),(1. / gamm));             

  r = rt / t;                                  

}










void hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  

  const double exner0 = 1.;    

  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    

  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); 

  p = p0 * pow(exner,(cp/rd));                                                         

  rt = pow((p / C0),(1. / gamm));                                                  

  r = rt / t;                                                                          

}








double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  

  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  

  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}



void finalize() {
  int ierr;
  free( state );
  free( state_tmp );
  free( flux );
  free( tend );
  free( hy_dens_cell );
  free( hy_dens_theta_cell );
  free( hy_dens_int );
  free( hy_dens_theta_int );
  free( hy_pressure_int );
  free( sendbuf_l );
  free( sendbuf_r );
  free( recvbuf_l );
  free( recvbuf_r );
  ierr = MPI_Finalize();
}




void reductions( double &mass , double &te ) {
  mass = 0;
  te   = 0;

  {
    for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      double r  =   state[ind_r] + hy_dens_cell[hs+k];             

      double u  =   state[ind_u] / r;                              

      double w  =   state[ind_w] / r;                              

      double th = ( state[ind_t] + hy_dens_theta_cell[hs+k] ) / r; 

      double p  = C0*pow(r*th,gamm);                               

      double t  = th / pow(p0/p,rd/cp);                            

      double ke = r*(u*u+w*w);                                     

      double ie = r*cv*t;                                          

      mass += r        *dx*dz; 

      te   += (ke + ie)*dx*dz; 

    }
  }
  }
  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}


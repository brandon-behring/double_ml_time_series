! Compile alongside the archived GetData2 and Constraint subroutines in an isolated directory.
program check_original
  implicit none
  integer :: totp,sbar,tbar,kidmax,ktp,nedraw,nmc,ftbar,vseed(3),vl,ncoeff,simul,nsimul
  double precision :: disc
  common /mydata/ totp,sbar,tbar,kidmax,ktp,nedraw,nmc,ftbar,vseed,vl,disc,ncoeff,simul,nsimul
  integer, parameter :: nfam=3201, maxkids=8
  integer :: rows(nfam*maxkids*2,26), nlines, i, sex, arm
  integer :: attendees(2,2), eligible(2,2)
  integer :: births(50), boys(50), school(50,maxkids), work(50,maxkids), dist(2)
  double precision :: yp(1), yc(1), tmp(1), gamv(20,3), bvec(5,3), subsidy(2)
  kidmax=maxkids
  call GetData2(nfam,rows,nlines)
  attendees=0
  eligible=0
  do i=1,nlines
    if (rows(i,18)/=98 .or. rows(i,20)/=1) cycle
    if (rows(i,10)<12 .or. rows(i,10)>15) cycle
    if (rows(i,13)<0 .or. rows(i,13)>1) cycle
    sex=rows(i,11)+1
    arm=rows(i,17)
    eligible(sex,arm)=eligible(sex,arm)+1
    attendees(sex,arm)=attendees(sex,arm)+rows(i,13)
  end do
  write(*,*) nlines, ((attendees(sex,arm),eligible(sex,arm),arm=1,2),sex=1,2)
  ! One child, observed attending after six completed grades; turn all other cash flows off.
  ftbar=50
  ktp=1
  gamv=0
  bvec=0
  dist=0
  births=0
  births(7)=1
  school=0
  school(14:20,1)=1
  work=0
  do sex=1,2
    boys=0
    boys(7)=sex-1
    call Constraint(yp,yc,tmp,20,0,births,boys,school,work,dist,gamv,bvec,40,1)
    subsidy(sex)=yc(1)
  end do
  write(*,*) subsidy
end program check_original

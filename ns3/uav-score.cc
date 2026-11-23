// ns-3 LTE scorer: one multi-UAV placement in, one score out.
//
// Reads a scenario file written by skycellbench/ns3.py (radio settings, UAV positions, users with
// their demands), builds an LTE network with an EPC core and one downlink UDP flow per user, and
// prints one JSON line: the score (mean share of demand delivered) and the per-user throughput.
//
// Compiled outside the ns-3 tree against an existing optimised ns-3.48 build (see build.sh).

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/propagation-module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace ns3;

// Mean air-to-ground path loss (Al-Hourani et al., 2014): free-space loss plus the expected
// excess loss, weighted by the line-of-sight probability at the link's elevation angle. It is a
// smooth function of position, the same formula as skycellbench/fast.py.
class A2gMeanLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("A2gMeanLossModel")
                .SetParent<PropagationLossModel>()
                .AddConstructor<A2gMeanLossModel>()
                .AddAttribute("Frequency", "Carrier frequency (Hz)", DoubleValue(2.12e9),
                              MakeDoubleAccessor(&A2gMeanLossModel::m_freq),
                              MakeDoubleChecker<double>())
                .AddAttribute("A", "Sigmoid parameter a", DoubleValue(9.61),
                              MakeDoubleAccessor(&A2gMeanLossModel::m_a),
                              MakeDoubleChecker<double>())
                .AddAttribute("B", "Sigmoid parameter b", DoubleValue(0.16),
                              MakeDoubleAccessor(&A2gMeanLossModel::m_b),
                              MakeDoubleChecker<double>())
                .AddAttribute("EtaLos", "Excess loss, line-of-sight (dB)", DoubleValue(1.0),
                              MakeDoubleAccessor(&A2gMeanLossModel::m_etaLos),
                              MakeDoubleChecker<double>())
                .AddAttribute("EtaNlos", "Excess loss, non-line-of-sight (dB)", DoubleValue(20.0),
                              MakeDoubleAccessor(&A2gMeanLossModel::m_etaNlos),
                              MakeDoubleChecker<double>());
        return tid;
    }

    double LossDb(const Vector& pa, const Vector& pb) const
    {
        const double dz = std::abs(pa.z - pb.z);
        const double r = std::hypot(pa.x - pb.x, pa.y - pb.y);
        const double d = std::max(1.0, std::hypot(r, dz));
        const double thetaDeg = std::atan2(dz, std::max(r, 1e-9)) * 180.0 / M_PI;
        const double pLos = 1.0 / (1.0 + m_a * std::exp(-m_b * (thetaDeg - m_a)));
        const double fspl = 20.0 * std::log10(4.0 * M_PI * m_freq * d / 299792458.0);
        return fspl + pLos * m_etaLos + (1.0 - pLos) * m_etaNlos;
    }

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override
    {
        return txPowerDbm - LossDb(a->GetPosition(), b->GetPosition());
    }

    int64_t DoAssignStreams(int64_t) override
    {
        return 0;
    }

    double m_freq;
    double m_a;
    double m_b;
    double m_etaLos;
    double m_etaNlos;
};

NS_OBJECT_ENSURE_REGISTERED(A2gMeanLossModel);

static std::vector<uint64_t> g_rxAtStart;

static void
Snapshot(ApplicationContainer* sinks)
{
    for (uint32_t i = 0; i < sinks->GetN(); ++i)
    {
        g_rxAtStart.push_back(sinks->Get(i)->GetObject<UdpServer>()->GetReceived());
    }
}

int
main(int argc, char* argv[])
{
    std::string scenario;
    double simTime = 2.0;  // s, measured window
    double attach = 0.5;   // s before traffic starts
    double settle = 0.3;   // s of traffic before the measured window
    uint32_t run = 1;
    std::string scheduler = "ns3::PfFfMacScheduler";

    CommandLine cmd(__FILE__);
    cmd.AddValue("scenario", "scenario file written by skycellbench", scenario);
    cmd.AddValue("simTime", "measured seconds", simTime);
    cmd.AddValue("run", "random stream run number", run);
    cmd.AddValue("scheduler", "MAC scheduler type", scheduler);
    cmd.Parse(argc, argv);

    std::ifstream in(scenario);
    if (!in)
    {
        std::cerr << "cannot read scenario file '" << scenario << "'" << std::endl;
        return 2;
    }
    std::string tag;
    double freq, txPower, noiseFigure, a, b, etaLos, etaNlos;
    uint32_t rb, nUav, nUe;
    in >> tag >> freq >> rb >> txPower >> noiseFigure >> a >> b >> etaLos >> etaNlos;
    in >> tag >> nUav;
    std::vector<Vector> uavPos(nUav);
    for (auto& p : uavPos)
    {
        in >> p.x >> p.y >> p.z;
    }
    in >> tag >> nUe;
    std::vector<Vector> uePos(nUe);
    std::vector<double> demand(nUe);
    for (uint32_t i = 0; i < nUe; ++i)
    {
        in >> uePos[i].x >> uePos[i].y >> demand[i];
        uePos[i].z = 1.5;
    }
    if (!in || tag != "users")
    {
        std::cerr << "malformed scenario file" << std::endl;
        return 2;
    }

    const auto t0 = std::chrono::steady_clock::now();
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(run);

    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(320));
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(txPower));
    Config::SetDefault("ns3::LteUePhy::NoiseFigure", DoubleValue(noiseFigure));
    // A user whose link is too weak simply receives nothing; it is not disconnected part-way
    // through the run, so the score does not depend on how long the simulation lasts.
    Config::SetDefault("ns3::LteUePhy::EnableRlfDetection", BooleanValue(false));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(0));
    Config::SetDefault("A2gMeanLossModel::Frequency", DoubleValue(freq));
    Config::SetDefault("A2gMeanLossModel::A", DoubleValue(a));
    Config::SetDefault("A2gMeanLossModel::B", DoubleValue(b));
    Config::SetDefault("A2gMeanLossModel::EtaLos", DoubleValue(etaLos));
    Config::SetDefault("A2gMeanLossModel::EtaNlos", DoubleValue(etaNlos));

    Ptr<UniformRandomVariable> uni = CreateObject<UniformRandomVariable>();

    NodeContainer enbNodes;
    NodeContainer ueNodes;
    enbNodes.Create(nUav);
    ueNodes.Create(nUe);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> pa = CreateObject<ListPositionAllocator>();
    for (const auto& p : uavPos)
    {
        pa->Add(p);
    }
    mob.SetPositionAllocator(pa);
    mob.Install(enbNodes);
    Ptr<ListPositionAllocator> pu = CreateObject<ListPositionAllocator>();
    for (const auto& p : uePos)
    {
        pu->Add(p);
    }
    mob.SetPositionAllocator(pu);
    mob.Install(ueNodes);

    Ptr<LteHelper> lte = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epc = CreateObject<PointToPointEpcHelper>();
    lte->SetEpcHelper(epc);
    lte->SetPathlossModelType(A2gMeanLossModel::GetTypeId());
    lte->SetSchedulerType(scheduler);
    lte->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(rb));
    lte->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(rb));
    NetDeviceContainer enbDevs = lte->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lte->InstallUeDevice(ueNodes);

    // Each user attaches to the UAV with the lowest path loss.
    Ptr<A2gMeanLossModel> probe = CreateObject<A2gMeanLossModel>();
    std::vector<uint32_t> serving(nUe, 0);
    for (uint32_t i = 0; i < nUe; ++i)
    {
        double bl = 1e30;
        for (uint32_t k = 0; k < nUav; ++k)
        {
            const double l = probe->LossDb(uavPos[k], uePos[i]);
            if (l < bl)
            {
                bl = l;
                serving[i] = k;
            }
        }
    }

    Ptr<Node> pgw = epc->GetPgwNode();
    NodeContainer remote;
    remote.Create(1);
    InternetStackHelper internet;
    internet.Install(remote);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
    NetDeviceContainer inet = p2p.Install(pgw, remote.Get(0));
    Ipv4AddressHelper ip;
    ip.SetBase("1.0.0.0", "255.0.0.0");
    ip.Assign(inet);
    Ipv4StaticRoutingHelper routing;
    routing.GetStaticRouting(remote.Get(0)->GetObject<Ipv4>())
        ->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIf = epc->AssignUeIpv4Address(ueDevs);

    const uint32_t pkt = 1200;
    ApplicationContainer sinks;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        routing.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
        lte->Attach(ueDevs.Get(i), enbDevs.Get(serving[i]));
        const double interval = pkt * 8.0 / (demand[i] * 1e6);
        UdpServerHelper server(9);
        sinks.Add(server.Install(ueNodes.Get(i)));
        UdpClientHelper client(ueIf.GetAddress(i), 9);
        client.SetAttribute("Interval", TimeValue(Seconds(interval)));
        client.SetAttribute("PacketSize", UintegerValue(pkt));
        ApplicationContainer c = client.Install(remote.Get(0));
        c.Start(Seconds(attach + uni->GetValue(0, interval)));
    }
    sinks.Start(Seconds(0.0));

    Simulator::Schedule(Seconds(attach + settle), &Snapshot, &sinks);
    Simulator::Stop(Seconds(attach + settle + simTime));
    Simulator::Run();

    std::vector<double> thr(nUe);
    double score = 0;
    double sum = 0;
    uint32_t connected = 0;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        const uint64_t got = sinks.Get(i)->GetObject<UdpServer>()->GetReceived() - g_rxAtStart[i];
        thr[i] = got * pkt * 8.0 / simTime / 1e6;
        score += std::min(1.0, thr[i] / demand[i]);
        sum += thr[i];
        if (ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetRrc()->GetState() ==
            LteUeRrc::CONNECTED_NORMALLY)
        {
            connected++;
        }
    }
    score /= nUe;
    Simulator::Destroy();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::ostringstream o;
    o.precision(8);
    o << "{\"score\":" << score << ",\"sum_thr_mbps\":" << sum << ",\"connected\":" << connected
      << ",\"sim_time\":" << simTime << ",\"run\":" << run << ",\"wall_s\":" << wall
      << ",\"thr_mbps\":[";
    for (uint32_t i = 0; i < nUe; ++i)
    {
        o << (i ? "," : "") << thr[i];
    }
    o << "],\"serving\":[";
    for (uint32_t i = 0; i < nUe; ++i)
    {
        o << (i ? "," : "") << serving[i];
    }
    o << "]}";
    std::cout << o.str() << std::endl;
    return 0;
}

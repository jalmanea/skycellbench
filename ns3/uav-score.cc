// ns-3 LTE scorer: one multi-UAV placement in, one score out.
//
// Reads a scenario file written by skycellbench/ns3.py (radio settings, UAV positions, users with
// their demands, and the path loss of every UAV-user link), builds an LTE network with an EPC core
// and one downlink UDP flow per user, and prints one JSON line: the score (mean share of demand
// delivered) and the per-user throughput.
//
// The path loss is not computed here. skycellbench computes one loss table per placement and both
// scorers use it; ns-3 takes it through its own MatrixPropagationLossModel, which LTE accepts
// like any other path loss model. Positions are passed for completeness and do not affect the
// radio.
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
    double txPower, noiseFigure;
    uint32_t rb, nUav, nUe;
    in >> tag >> rb >> txPower >> noiseFigure;
    if (tag != "radio2")
    {
        std::cerr << "scenario file is not in the loss-table format" << std::endl;
        return 2;
    }
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
        in >> uePos[i].x >> uePos[i].y >> uePos[i].z >> demand[i];
    }
    std::string tagLoss;
    in >> tagLoss;
    std::vector<std::vector<double>> lossDb(nUav, std::vector<double>(nUe));
    for (auto& row : lossDb)
    {
        for (auto& v : row)
        {
            in >> v;
        }
    }
    if (!in || tag != "users" || tagLoss != "loss")
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
    lte->SetPathlossModelType(MatrixPropagationLossModel::GetTypeId());
    lte->SetSchedulerType(scheduler);
    lte->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(rb));
    lte->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(rb));
    NetDeviceContainer enbDevs = lte->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lte->InstallUeDevice(ueNodes);

    // The helper has created one loss model per direction; fill both from the table. Only
    // UAV-user pairs ever exchange signals, so every other pair keeps a loss no signal survives.
    for (const auto& channel : {lte->GetDownlinkSpectrumChannel(), lte->GetUplinkSpectrumChannel()})
    {
        Ptr<MatrixPropagationLossModel> m =
            DynamicCast<MatrixPropagationLossModel>(channel->GetPropagationLossModel());
        NS_ABORT_MSG_IF(!m, "the LTE channel does not use the matrix loss model");
        m->SetDefaultLoss(400.0);
        for (uint32_t k = 0; k < nUav; ++k)
        {
            Ptr<MobilityModel> a = enbNodes.Get(k)->GetObject<MobilityModel>();
            for (uint32_t i = 0; i < nUe; ++i)
            {
                m->SetLoss(a, ueNodes.Get(i)->GetObject<MobilityModel>(), lossDb[k][i]);
            }
        }
    }

    // Each user attaches to the UAV with the lowest path loss (the first one on a tie).
    std::vector<uint32_t> serving(nUe, 0);
    for (uint32_t i = 0; i < nUe; ++i)
    {
        double bl = 1e30;
        for (uint32_t k = 0; k < nUav; ++k)
        {
            const double l = lossDb[k][i];
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


# Host

## Anforderung

* Auf externem Speicher vorliegende Daten sollen in einen Puffer geladen werden. Zwecks Weiterverarbeitung muss eine Schnittstelle bereitgestellt werden, um diese Daten effektiv aus dem Puffer in den Graphikkartenspeicher zu transportieren.
* Das Auslesen aus dem Puffer erfolgt durch mehrere Threads. Alle Operationen auf dem Puffer müssen demnach threadsafe gestaltet sein.

## Hauptbestandteile

Die Anforderungen sind typisch für ein Erzeuger-Verbraucher-Problem. Dieses wird durch folgende Komponenten gelöst:

### Ringpuffer-Klasse

Der Ringpuffer wird durch Erzeuger und Verbraucher gefüllt/geleert. Die Puffergröße ist fix. Threadsicherheit wird mittels Semaphoren in die Ringpufferklasse implementiert. Da momentan POSIX-Plattformen als Ziel genannt sind, wird `semaphore.h` dafür genutzt.

\begin{lstlisting}
template <class type>
class Ringbuffer() {
private:
	sem_t mtx;
	sem_t full, empty;
public:
    Ringbuffer(size_t bSize);
    ~Ringbuffer();
    type* reserveHead(unsigned int count);
    int freeHead(type* data, unsigned int count);
    type* reserveTail(unsigned int count);
    int freeTail(type* gpu_data, unsigned int count);
}	
\end{lstlisting}
            
            
+ Der Zustand voll / leer wird über die Semaphore `sem_w` (Anfangszustand `bSize`), `sem_r` (Anfangszustand `0`) festgestellt. 
+ Die Funktion `write` blockiert bei vollem Puffer um ein Verlust an Messdaten entgegenzuwirken. Für späteres Lesen eines Datenstroms direkt vom Messgerät kann auch nichtblockierendes Schreiben in den Puffer mit entsprechendem Überlauf implementiert werden.
+ `readToCUDA` kopiert die Daten vom Tail des Puffers nach `*gpu_data`. Die Verantwortlichkeit für `cudaSetDevice(deviceId)` ist noch zu klären. Der Kopiervorgang kann nicht asynchron ablaufen, um den Speicher im Puffer sicher freigeben zu können.
+ `readToHost` kopiert die Daten vom Tail des Puffers an eine Hostspeicheradresse. Dies dient zum Testen der CPU-Variante des Auswertealgorithmus.
+ `size` in `write`, `readToHost` und `readToCUDA` gibt jeweils die Anzahl der auszutauschenden Waveforms vom Datentyp `wform` an.

3 Semaphoren:

+ `sem_mtx` für exklusiven Zugriff auf Speicher
+ `sem_r` um Lesezugriff bei leerem Puffer zu regeln
+ `sem_w` um Schreibzugriff bei vollem Puffer zu regeln

Der Datentyp `wform` ist zunächst fix. In einer zweiten Version wird er per Template implementiert.

### Erzeuger-Klasse

Verantwortlich für das Einlesen der Daten aus einer Verzeichnisstruktur in den Puffer. Die Waveforms aus den beiden Kanälen werden aufgetrennt in zwei Waveforms. Dazu wird ein kleiner interner Puffer verwendet.
Der Erzeuger versucht zu schreiben, solange er Daten liefern kann. Der tatsächliche Schreibvorgang wird über das Semaphore `sem_w` innerhalb der Ringpufferklasse gesteuert.

Die sample-Daten sind laut Code vom Typ `short int`. Damit ergibt sich für `wform`

    typedef short int wform[nSample]
  
### Verbraucher

Der Verbraucher wird im Teil "Speichertransfer" beschrieben. Das Semaphore `sem_r` regelt den Lesezugriff innerhalb der Ringpufferklasse.

Der Verbraucher stellt über entsprechendes mapping sicher, dass die Datenreihenfolge bekannt bleibt, um die Ergebnisse den Eingangsdaten zuordnen zu können.


## Tasks

* Erzeugerklasse: Lesen der Messdaten mit root-Verzeichnis der Daten als Eingabewert
* erwartete Ordnung der Messdaten nach Einlesen im Puffer beschreiben
* Ringpuffer initialisieren
    * Speicherreservierung
    * Semaphoren: `sem_mtx`, `sem_w`, `sem_r`
* Ringpuffer: `write`
* Ringpuffer: `readToHost`
* Ringpuffer: `readToCUDA`
* Vergleich der Ergebnisse CPU vs GPU-Auswertung
* Skalierung
* Template-version des Ringpuffers



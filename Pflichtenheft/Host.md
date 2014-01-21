
# Host

## Anforderung

* Auf externem Speicher vorliegende Daten sollen in einen Puffer geladen werden. Zwecks Weiterverarbeitung muss eine Schnittstelle bereitgestellt werden, um diese Daten effektiv aus dem Puffer in den Grafikkartenspeicher zu transportieren.
* Das Auslesen aus dem Puffer erfolgt durch mehrere Threads. Alle Operationen auf dem Puffer müssen demnach threadsafe gestaltet sein.

## Hauptbestandteile

Die Anforderungen sind typisch für ein Erzeuger-Verbraucher-Problem. Dieses wird durch folgende Komponenten gelöst:

### Ringpuffer-Klasse

Der Ringpuffer wird durch Erzeuger und Verbraucher gefüllt/geleert. Die Puffergröße ist fix. Threadsicherheit wird mittels Semaphoren in die Ringpufferklasse implementiert. Da momentan POSIX-Plattformen als Ziel genannt sind, wird `semaphore.h` dafür genutzt.

\begin{lstlisting}
template <class Type>
class Ringbuffer() {
private:
	sem_t mtx;
	sem_t full, empty;
public:
    Ringbuffer(unsigned int bSize);
    ~Ringbuffer();
    int writeFromHost(Type* inputOnHost);
    int copyToHost(Type* outputOnHost);
    Type* reserveHead();
    int freeHead();
    Type* reserveTail();
    int freeTail();
    bool doQuit();
}	
\end{lstlisting}
            
            
+ `int writeFromHost(Type* inputOnHost)` ermöglicht das Schreiben der Daten
`inputOnHost` in den Puffer. Dazu müssen die Daten im Speicher des Host liegen.
Der Rückgabewerte ist `0` wenn das Schreiben erfolgreich war, sonst eine
positive Zahl. Bei vollem Puffer blockiert der Aufruf bis wieder Platz
verfügbar ist.
+ `int copyToHost(Type* outputOnHost)` liest Daten aus dem Puffer nach
`outputOnHost`. Bei leerem Puffer blockiert der Aufruf bis Daten im Puffer
verfügbar sind. Der Zeiger `outputOnHost` muss auf eine Hostaddresse zeigen.
+ Um Daten von der Grafikkarte in den Puffer zu schreiben, muss der 
Kopiervorgang vom Aufrufer selbst durchgeführt werden. Dazu kann vom Puffer
mit `Type* reserveHead()` eine verfügbare Speicheraddresse angefordert werden.
Sollte kein Platz im Puffer sein, blockiert der Aufruf bis dies der Fall ist.
Bis zum Aufruf von `int freeHead()` ist der Puffer für andere Aktivitäten 
blockiert.
+ `Type* reserveTail()` ermöglicht Daten aus dem Puffer auf die Grafikkarte
zu laden. Die Rückgabeaddresse ist die reservierte Addresse im Puffer aus
der die Daten vom Aufrufer selbst kopiert werden können. Die Addresse und
der Puffer müssen mit `int freeTail()` wieder freigegeben werden.
+ `freeHead()` und `freeTail()` geben im Erfolgsfall `0` zurück. Bei einem 
Fehler ist der Rückgabewert positiv.
+ Ist der Puffer leer und hat der Erzeuger gemeldet, dass keine Daten mehr
geliefert werden, so gibt der Puffer auf `bool doQuit()` `true` zurück. 
Andernfalls `false`.

### Erzeuger-Klasse

Verantwortlich für das Einlesen der Daten aus einer Verzeichnisstruktur in den 
Puffer. Die Waveforms aus den beiden Kanälen werden aufgetrennt in zwei 
Waveforms. Dazu wird ein kleiner interner Puffer verwendet.
Der Erzeuger versucht zu schreiben, solange er Daten liefern kann. Der 
tatsächliche Schreibvorgang wird über das Semaphore `sem_w` innerhalb der Ringpufferklasse gesteuert.

Die sample-Daten sind laut Code vom Typ `short int`. Folgende Typen werden 
verwendet

    typedef short int sample_t              // Samplingpunkte
    
    typedef std::vector<sample_t> wform_t
    wform_t.reserve(SAMPLE_COUNT)           // eine waveform
    
    typedef std::vector<wform_t> chunk_t
    chunk_t.reserve(CHUNK_COUNT)            // Ein Chunk waveforms
  
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


